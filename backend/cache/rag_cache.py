"""RAG 检索证据语义缓存（只缓存检索证据，不缓存最终答案）。

命中时跳过 Multi-Query（3 次 LLM 改写）+ 混合检索（最多 4 次 embedding）+ RRF + Rerank
整段检索管线，但仍由 LLM 用当前会话上下文生成答案，保证正确性（不跨会话复用答案）。

两层结构：
- 精确层：normalize(query) → sha1 → Redis string key `rag_cache:v{ver}:{hash}`（0 次 embedding）。
  Redis 不可用时自动降级到进程内 dict（沿用 redis_client 的降级哲学）。
- 语义层：embed(query) → 与进程内 {query向量 → 结果} 条目余弦相似度，max ≥ threshold 命中。
  纯内存、无持久化、重启自热（首次命中重跑管线重填）。

失效机制：缓存条目都带 version = knowledge_indexer.status().last_built_at（epoch 秒）。
rebuild 后 last_built_at 变化 → 旧精确 key 成孤儿自然 TTL 过期；语义条目版本不匹配时惰性丢弃。
无需改动 indexer.rebuild_index()。

配置项（config.py Settings，env 可覆盖，改后需重启进程，见 CLAUDE.md 4.4）：
  RAG_CACHE_ENABLED / RAG_CACHE_TTL / RAG_CACHE_THRESHOLD / RAG_CACHE_MAX_ENTRIES
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import asdict
from typing import Any

from cache.redis_client import redis_client
from config import get_settings
from embeddings_client import EmbeddingClient
from knowledge_retrieval.indexer import knowledge_indexer
from knowledge_retrieval.types import (
    Evidence,
    OrchestratedRetrievalResult,
    RetrievalStep,
)

logger = logging.getLogger(__name__)

# ---------- 查询规范化（精确层专用，去语气词/标点，纯正则无 NLP） ----------

_FILLER_WORDS = ("请问一下", "请帮我", "请问", "帮我", "麻烦", "你好", "请", "我想要")
_PUNCT_RE = re.compile(r"[^\w一-鿿\s]")
_SPACE_RE = re.compile(r"\s+")


def _normalize_query(query: str) -> str:
    s = query.lower().strip()
    for word in _FILLER_WORDS:
        s = s.replace(word, "")
    s = _PUNCT_RE.sub(" ", s)
    return _SPACE_RE.sub(" ", s).strip()


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（手写，避免引入 numpy 依赖；几百条 × 1536 维为微秒级）。"""
    if not a or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------- 结果序列化（只读 types.py，不加字段，符合契约规则 4.1） ----------


def _result_to_dict(result: OrchestratedRetrievalResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "evidences": [asdict(e) for e in result.evidences],
        "steps": [step.to_dict() for step in result.steps],
        "reason": result.reason,
    }


def _result_from_dict(data: dict[str, Any]) -> OrchestratedRetrievalResult:
    return OrchestratedRetrievalResult(
        status=data["status"],
        evidences=[Evidence(**e) for e in data.get("evidences", [])],
        steps=[
            RetrievalStep(
                kind=step["kind"],
                stage=step["stage"],
                title=step["title"],
                message=step.get("message", ""),
                results=[Evidence(**r) for r in step.get("results", [])],
            )
            for step in data.get("steps", [])
        ],
        reason=data.get("reason", ""),
    )


class RagCache:
    """检索证据两层语义缓存（Redis 精确层 + 内存语义层），Redis 不可用时自动降级。"""

    _CACHEABLE_STATUSES = {"success"}  # orchestrator 现只产出 success/not_found（窄路径已移除）

    def __init__(self) -> None:
        self._embedder: EmbeddingClient | None = None
        self._exact: dict[str, dict[str, Any]] = {}  # Redis 不可用时的内存兜底 {key: {data, stored_at}}
        self._entries: list[dict[str, Any]] = []  # 语义层条目 {vec, norm, result, version, stored_at}
        self._last_norm: str | None = None  # 最近一次 embedding 的规范化 query（避免 get/put 重复嵌入）
        self._last_vec: list[float] | None = None

    @property
    def enabled(self) -> bool:
        return get_settings().rag_cache_enabled

    # ---------- 内部工具 ----------

    @staticmethod
    def _index_ready() -> bool:
        return knowledge_indexer.status().ready

    @staticmethod
    def _version() -> float:
        return knowledge_indexer.status().last_built_at or 0.0

    def _embed_cached(self, query: str) -> list[float] | None:
        """embedding 复用：同一请求内 get → put 两次调用不重复打接口。"""
        norm = _normalize_query(query)
        if self._last_norm == norm and self._last_vec is not None:
            return self._last_vec
        if self._embedder is None:
            self._embedder = EmbeddingClient()
        try:
            vec = self._embedder.embed_one(query)
        except Exception:
            logger.warning("[rag_cache] embedding 失败，语义层跳过", exc_info=True)
            self._last_norm = norm
            self._last_vec = None
            return None
        self._last_norm = norm
        self._last_vec = vec
        return vec

    # ---------- 精确层（Redis + 内存兜底） ----------

    @staticmethod
    def _exact_key(version: float, norm: str) -> str:
        digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        return f"rag_cache:v{version}:{digest}"

    def _exact_get(self, key: str) -> OrchestratedRetrievalResult | None:
        raw = redis_client.get(key)
        if raw is not None:
            try:
                return _result_from_dict(json.loads(raw))
            except Exception:
                return None
        entry = self._exact.get(key)
        if entry is None:
            return None
        ttl = get_settings().rag_cache_ttl
        if time.time() - entry["stored_at"] > ttl:
            self._exact.pop(key, None)
            return None
        try:
            return _result_from_dict(entry["data"])
        except Exception:
            return None

    def _exact_set(self, key: str, result_dict: dict[str, Any]) -> None:
        raw = json.dumps(result_dict, ensure_ascii=False)
        ttl = get_settings().rag_cache_ttl
        if not redis_client.set(key, raw, ttl=ttl):
            # Redis 不可用：写入内存兜底，并惰性清理过期项
            self._exact[key] = {"data": result_dict, "stored_at": time.time()}
            now = time.time()
            self._exact = {
                k: v for k, v in self._exact.items() if now - v["stored_at"] <= ttl
            }

    # ---------- 语义层（进程内内存，无持久化） ----------

    def _evict_stale(self, version: float) -> None:
        ttl = get_settings().rag_cache_ttl
        now = time.time()
        self._entries = [
            e for e in self._entries
            if e["version"] == version and now - e["stored_at"] <= ttl
        ]

    def _semantic_get(self, query: str, version: float) -> OrchestratedRetrievalResult | None:
        self._evict_stale(version)
        if not self._entries:
            return None
        qvec = self._embed_cached(query)
        if qvec is None:
            return None
        threshold = get_settings().rag_cache_threshold
        best_entry: dict[str, Any] | None = None
        best_sim = 0.0
        for entry in self._entries:
            sim = _cosine(qvec, entry["vec"])
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
        if best_entry is None or best_sim < threshold:
            return None
        try:
            return _result_from_dict(best_entry["result"])
        except Exception:
            return None

    def _semantic_put(self, query: str, result_dict: dict[str, Any], version: float) -> None:
        qvec = self._embed_cached(query)
        if qvec is None:
            return
        self._evict_stale(version)
        self._entries.append(
            {
                "vec": qvec,
                "norm": _normalize_query(query),
                "result": result_dict,
                "version": version,
                "stored_at": time.time(),
            }
        )
        max_entries = get_settings().rag_cache_max_entries
        if len(self._entries) > max_entries:
            # 超上限淘汰最旧
            self._entries.sort(key=lambda e: e["stored_at"])
            self._entries = self._entries[-max_entries:]

    # ---------- 对外 API ----------

    def get(self, query: str) -> OrchestratedRetrievalResult | None:
        """查询缓存。未启用 / 索引未就绪 / 未命中返回 None（等价于照常跑检索管线）。"""
        if not self.enabled or not query:
            return None
        if not self._index_ready():
            return None
        version = self._version()
        key = self._exact_key(version, _normalize_query(query))

        cached = self._exact_get(key)
        if cached is not None:
            logger.info("[rag_cache] 精确命中 version=%s query=%r", version, query)
            return cached

        cached = self._semantic_get(query, version)
        if cached is not None:
            logger.info("[rag_cache] 语义命中 version=%s query=%r", version, query)
            # 语义命中后回填精确层：下次同款提问走 0 embedding 的精确路径
            self._exact_set(key, _result_to_dict(cached))
            return cached

        return None

    def put(self, query: str, result: OrchestratedRetrievalResult) -> None:
        """写入缓存。仅缓存 status 为 success/partial 且 evidence 非空的结果。"""
        if not self.enabled or not query:
            return
        if result.status not in self._CACHEABLE_STATUSES or not result.evidences:
            return
        result_dict = _result_to_dict(result)
        version = self._version()
        self._exact_set(self._exact_key(version, _normalize_query(query)), result_dict)
        self._semantic_put(query, result_dict, version)


rag_cache = RagCache()
