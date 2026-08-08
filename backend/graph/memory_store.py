"""简单长期记忆：纯 ChromaDB 嵌入式 SQLite（单 collection + 应用侧阈值）。

一个 `memories` collection（documents=text，embeddings=向量，metadata 存 created_at）：
- 写入：一次 LLM 抽取对话中的稳定事实 → 批量嵌入 → 与已有记忆余弦去重（> DEDUP_THRESHOLD 跳过）→ add；
- 读取：嵌入 query → collection.query 余弦距离 → 转相似度 → 阈值过滤 → top_k。
无 PostgreSQL / 无 psycopg2 / 无 numpy 全表扫描 / 无 scope/category/status 枚举 / 无整合决策器 / 无遗忘规则（仅行数上限裁剪 `_prune`）。

存储目录 `storage/memory/chroma/`，与知识库 `storage/knowledge/vector/chroma/` 完全隔离
（知识库 rebuild 只 delete 它自己目录里的 collection，物理上碰不到这里）。

ChromaDB 不可用（未配置 / 初始化失败）→ is_ready()=False，检索自动降级（同 redis 降级模式）。
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
from chromadb.config import Settings as ChromaSettings

from langsmith import traceable

from config import get_settings
from embeddings_client import EmbeddingClient

logger = logging.getLogger(__name__)

# 检索相关性阈值：低于该分数视为不相关，不注入上下文（相关性门槛）
MIN_SCORE = 0.4
# 写入去重余弦阈值：与已有记忆相似度超过该值视为重复，跳过（近似语义去重）
DEDUP_THRESHOLD = 0.93
# 行数上限：超限删最旧（锁死 retrieve / 去重的扫描边界）
MAX_MEMORIES = 2000
# retrieve 单次 query 的 n_results 上限（防御性，MAX_MEMORIES 下永不触达）
RETRIEVE_POOL = 3000
# 抽取器：单次输入的最大消息条数
EXTRACT_MESSAGES_MAX = 20

# Chroma collection 名与持久化目录（相对 base_dir，与知识库 chroma 目录隔离）
COLLECTION_NAME = "memories"
COLLECTION_DIR = "storage/memory/chroma"

# ---------- 抽取器提示词（简化版：只输出纯 text 数组，去 scope/category/confidence） ----------

EXTRACTOR_SYSTEM_PROMPT = (
    "# 角色\n"
    "你是长期记忆抽取器，从对话中提炼值得跨会话记住的稳定事实。\n"
    "\n"
    "# 规则\n"
    "1. 只抽「稳定、可复用」的信息：\n"
    "   - 用户偏好（语言、格式、工作方式、工具偏好）\n"
    "   - 个人属性（职业、技能、角色）\n"
    "   - 项目与领域事实（项目背景、技术选型、业务规则）\n"
    "   - 明确的决策与目标\n"
    "   - 长期约束（红线、必须遵守的规则）\n"
    "2. 不抽：寒暄、一次性请求、纯提问、情绪、临时的偶然信息。\n"
    "3. 一条记忆 = 一个原子陈述，用陈述句、第三人称，不含多个要点。\n"
    "4. 没有值得记住的内容 → 返回空数组。\n"
    "\n"
    "# 输出（严格 JSON，不要任何多余文字；每项只含 text 字符串）\n"
    '{"memories": ["User 偏好英文交流", "项目后端用 PostgreSQL 存长期记忆"]}\n'
)


class MemoryStore:
    def __init__(self) -> None:
        self.base_dir: Path | None = None
        self._client: chromadb.ClientAPI | None = None
        self._collection: Any | None = None

    # ---------- 生命周期 ----------

    def configure(self, base_dir: Path) -> None:
        """初始化 ChromaDB 持久化客户端 + collection。失败 → 降级（is_ready()=False）。"""
        self.base_dir = base_dir
        try:
            path = base_dir / COLLECTION_DIR
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"[memory_store] ChromaDB 长期记忆就绪：{path}")
        except Exception:
            logger.warning("[memory_store] ChromaDB 初始化失败，长期记忆降级关闭", exc_info=True)
            self._client = None
            self._collection = None

    def is_ready(self) -> bool:
        """ChromaDB collection 可用即就绪（嵌入失败由 retrieve/remember 各自捕获）。"""
        return self._collection is not None

    def status(self) -> dict[str, Any]:
        count = 0
        if self._collection is not None:
            try:
                count = self._collection.count()
            except Exception:
                pass
        return {"ready": self.is_ready(), "count": count}

    # ---------- 读取 ----------

    @traceable(
        run_type="retriever",
        name="memory_retrieve",
        process_outputs=lambda o: {"n": len(o)},
    )
    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """语义召回：嵌入 query → collection.query 余弦距离 → 阈值过滤 + 排序。

        返回 `[{text, score, source}]`，与旧实现字段兼容（source 恒为 "memory"）。
        """
        if not self.is_ready() or not query.strip():
            return []
        try:
            if self._collection.count() == 0:
                return []
            q = EmbeddingClient().embed_one(query)
            res = self._collection.query(
                query_embeddings=[q],
                n_results=RETRIEVE_POOL,
                include=["documents", "metadatas", "distances"],
            )
            docs = (res.get("documents") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
        except Exception:
            logger.warning("[memory_store] 向量召回失败", exc_info=True)
            return []

        hits: list[dict[str, Any]] = []
        for doc, dist in zip(docs, dists):
            score = 1.0 - float(dist)  # Chroma 余弦距离 ∈ [0,2]，相似度 = 1 - distance
            if score < MIN_SCORE:
                continue
            hits.append(
                {
                    "text": str(doc or ""),
                    "score": round(score, 4),
                    "source": "memory",
                }
            )
        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[:top_k]

    # ---------- 写入：自动抽取（一次 LLM 调用） ----------

    async def remember(self, messages: list[dict[str, Any]]) -> int:
        """从会话消息自动写入长期记忆：LLM 抽取 → 批量嵌入 → 余弦去重 → add。

        返回实际写入条数。任何一步失败都不抛出，只打日志（调用方是后台任务，不能影响 SSE 响应）。
        """
        if not self.is_ready():
            return 0
        # 只保留 user/assistant 且有内容的最近消息（丢弃空内容 / 工具调用）
        recent: list[dict[str, str]] = []
        for m in messages:
            if m.get("role") not in {"user", "assistant"}:
                continue
            content = str(m.get("content", "") or "").strip()
            if content:
                recent.append({"role": m["role"], "content": content})
        if len(recent) < 2:
            return 0
        recent = recent[-EXTRACT_MESSAGES_MAX:]

        facts = await self._extract_facts(recent)
        if not facts:
            return 0
        try:
            embeddings = EmbeddingClient().embed(facts)
        except Exception:
            logger.warning("[memory_store] 记忆嵌入失败", exc_info=True)
            return 0
        if len(embeddings) != len(facts):
            return 0

        written = self._insert_new(list(zip(facts, embeddings)))
        if written:
            self._prune()
        return written

    async def _extract_facts(self, messages: list[dict[str, str]]) -> list[str]:
        """抽取器：对话 → 纯 text 事实列表。解析失败 / 空数组 → []。"""
        prompt = (
            EXTRACTOR_SYSTEM_PROMPT
            + "\n\n# 对话记录\n"
            + json.dumps(messages, ensure_ascii=False)
        )
        try:
            response = await self._build_llm().ainvoke(
                [{"role": "system", "content": prompt}]
            )
        except Exception:
            logger.warning("[memory_store] 记忆抽取 LLM 调用失败", exc_info=True)
            return []
        data = self._parse_json_block(_llm_text(getattr(response, "content", "")))
        if not data:
            return []

        facts: list[str] = []
        seen: set[str] = set()
        for item in data.get("memories") or []:
            # 兼容 str 或旧版 {"text": ...} 两种输出
            text = (
                str(item).strip()
                if isinstance(item, str)
                else str((item or {}).get("text", "") or "").strip()
            )
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            facts.append(text)
        return facts

    # ---------- 内部实现 ----------

    def _insert_new(self, items: list[tuple[str, list[float]]]) -> int:
        """批量写入：逐条与已有记忆余弦去重（> DEDUP_THRESHOLD 跳过）→ add。返回实际写入条数。"""
        if not items:
            return 0
        written = 0
        for text, vector in items:
            # 去重：查与已有记忆最相似的一条，相似度超阈值即视为重复
            sim = 0.0
            try:
                dup = self._collection.query(
                    query_embeddings=[vector],
                    n_results=1,
                    include=["distances"],
                )
                d = (dup.get("distances") or [[]])[0]
                if d:
                    sim = 1.0 - float(d[0])
            except Exception:
                logger.warning("[memory_store] 记忆去重查询失败", exc_info=True)
            if sim > DEDUP_THRESHOLD:
                continue
            self._collection.add(
                ids=[str(uuid4())],
                documents=[text],
                embeddings=[vector],
                metadatas=[{"created_at": time.time()}],
            )
            written += 1
        return written

    def _prune(self) -> None:
        """行数上限：超限删 created_at 最早的。失败静默。"""
        if self._collection is None:
            return
        try:
            total = self._collection.count()
            if total > MAX_MEMORIES:
                to_delete = total - MAX_MEMORIES
                res = self._collection.get(include=["metadatas"])
                ids = res.get("ids") or []
                metas = res.get("metadatas") or []
                pairs = sorted(
                    ((m.get("created_at") or 0.0, i) for m, i in zip(metas, ids))
                )
                oldest_ids = [i for _, i in pairs[:to_delete]]
                self._collection.delete(ids=oldest_ids)
        except Exception:
            logger.warning("[memory_store] 记忆裁剪失败", exc_info=True)

    def _build_llm(self) -> Any:
        """记忆抽取专用轻量 LLM：优先 SUMMARY_MODEL（便宜），未配置回退主模型。"""
        settings = get_settings()
        from langchain_openai import ChatOpenAI

        if settings.summary_api_key:
            return ChatOpenAI(
                model=settings.summary_model,
                api_key=settings.summary_api_key,
                base_url=settings.summary_base_url,
                temperature=0,
            )
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=0,
        )

    @staticmethod
    def _parse_json_block(text: str) -> dict[str, Any] | None:
        """从 LLM 输出中稳健提取 JSON 对象（容忍 markdown 围栏 / 前后废话）。"""
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


def _llm_text(content: Any) -> str:
    """把 LLM 返回的 content（str 或 OpenAI 多模态 block 列表）归一为纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


memory_store = MemoryStore()
