"""简单长期记忆：PostgreSQL 单表 + 应用侧余弦召回。

一张 `memories` 表（text + embedding float8[]）：
- 写入：一次 LLM 抽取对话中的稳定事实 → 批量嵌入 → 与已有记忆余弦去重（> DEDUP_THRESHOLD 跳过）→ INSERT；
- 读取：嵌入 query → 全表向量 numpy 余弦 → 阈值过滤 → top_k → 更新 last_used_at。
无 ChromaDB、无 scope/category/status 枚举、无整合决策器、无遗忘规则（仅行数上限裁剪 `_prune`）。

PG 不可用（未配置 / 连接失败）→ is_ready()=False，检索自动降级（同 redis 降级模式）。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from langsmith import traceable

from config import get_settings
from embeddings_client import EmbeddingClient

logger = logging.getLogger(__name__)

# 检索相关性阈值：低于该分数视为不相关，不注入上下文（相关性门槛）
MIN_SCORE = 0.4
# 写入去重余弦阈值：与已有记忆相似度超过该值视为重复，跳过（近似语义去重）
DEDUP_THRESHOLD = 0.93
# 行数上限：超限删最旧（锁死 retrieve / 去重的全表扫描边界）
MAX_MEMORIES = 2000
# retrieve 全表 SELECT 上限（防御性，MAX_MEMORIES 下永不触达）
RETRIEVE_POOL = 3000
# 抽取器：单次输入的最大消息条数
EXTRACT_MESSAGES_MAX = 20

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
        self._pool: Any = None  # psycopg2 SimpleConnectionPool

    # ---------- 生命周期 ----------

    def configure(self, base_dir: Path) -> None:
        """初始化 PG 连接池 + 建表。PG 不可用 → 降级（is_ready()=False）。"""
        self.base_dir = base_dir
        dsn = get_settings().pg_dsn
        if not dsn:
            logger.info("[memory_store] PG_DSN 未配置，长期记忆存储降级关闭")
            self._pool = None
            return
        try:
            import psycopg2.pool

            self._pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1, maxconn=5, dsn=dsn, connect_timeout=5
            )
            self._ensure_schema()
            logger.info("[memory_store] PostgreSQL 长期记忆就绪")
        except Exception:
            logger.warning("[memory_store] PostgreSQL 连接失败，长期记忆存储降级", exc_info=True)
            self._pool = None

    def is_ready(self) -> bool:
        """PG 连接池可用即就绪（不再依赖 embedding key，嵌入失败由 retrieve/remember 各自捕获）。"""
        return self._pool is not None

    def status(self) -> dict[str, Any]:
        count = 0
        if self._pool is not None:
            try:
                conn = self._pool.getconn()
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT count(*) FROM memories")
                        count = int(cur.fetchone()[0])
                finally:
                    self._pool.putconn(conn)
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
        """语义召回：嵌入 query → 全表向量 numpy 余弦 → 阈值过滤 + 排序。

        返回 `[{text, score, source}]`，与旧实现字段兼容（source 恒为 "memory"）。
        """
        if not self.is_ready() or not query.strip():
            return []
        try:
            q = np.asarray(EmbeddingClient().embed_one(query), dtype=np.float32)
            rows = self._load_all()
            if not rows:
                return []
            mat = np.stack([np.asarray(r["embedding"], dtype=np.float32) for r in rows])
            q_norm = np.linalg.norm(q)
            norms = np.linalg.norm(mat, axis=1)
            scores = (mat @ q) / (norms * q_norm + 1e-9)
        except Exception:
            logger.warning("[memory_store] 向量召回失败", exc_info=True)
            return []

        hits: list[tuple[int, dict[str, Any]]] = []
        for i, row in enumerate(rows):
            score = float(scores[i])
            if score < MIN_SCORE:
                continue
            hits.append(
                (
                    row["id"],
                    {
                        "text": row["text"],
                        "score": round(score, 4),
                        "source": "memory",
                    },
                )
            )
        hits.sort(key=lambda item: item[1]["score"], reverse=True)
        hits = hits[:top_k]
        if hits:
            self._touch([item[0] for item in hits])
        return [item[1] for item in hits]

    # ---------- 写入：自动抽取（一次 LLM 调用） ----------

    async def remember(self, messages: list[dict[str, Any]]) -> int:
        """从会话消息自动写入长期记忆：LLM 抽取 → 批量嵌入 → 余弦去重 → INSERT。

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

    def _ensure_schema(self) -> None:
        """建 memories 表；旧结构（无 embedding 列）→ DROP 重建。幂等。"""
        if self._pool is None:
            return
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                # 旧表有 scope/category/status/metadata 而无 embedding 列，向量无法使用 → 丢弃重建
                cur.execute("SELECT to_regclass('memories')")
                if cur.fetchone()[0] is not None:
                    cur.execute(
                        "SELECT count(*) FROM information_schema.columns "
                        "WHERE table_name = 'memories' AND column_name = 'embedding'"
                    )
                    if int(cur.fetchone()[0]) == 0:
                        cur.execute("SELECT count(*) FROM memories")
                        n = int(cur.fetchone()[0])
                        logger.warning(
                            "[memory_store] 旧 memories 表缺 embedding 列，DROP 重建（丢弃 %s 条存量）", n
                        )
                        cur.execute("DROP TABLE memories")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id           BIGSERIAL PRIMARY KEY,
                        text         TEXT NOT NULL,
                        embedding    FLOAT8[] NOT NULL,
                        created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                        last_used_at TIMESTAMPTZ
                    )
                    """
                )
            conn.commit()
        finally:
            self._pool.putconn(conn)

    def _load_all(self) -> list[dict[str, Any]]:
        """取全表 (id, text, embedding)，按 id 倒序，上限 RETRIEVE_POOL。失败返回 []。"""
        if self._pool is None:
            return []
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, text, embedding FROM memories ORDER BY id DESC LIMIT %s",
                    (RETRIEVE_POOL,),
                )
                return [
                    {"id": int(row[0]), "text": row[1], "embedding": row[2]}
                    for row in cur.fetchall()
                ]
        except Exception:
            logger.warning("[memory_store] PG 全表读取失败", exc_info=True)
            return []
        finally:
            self._pool.putconn(conn)

    def _insert_new(self, items: list[tuple[str, list[float]]]) -> int:
        """批量写入：与已有记忆余弦去重（> DEDUP_THRESHOLD 跳过）→ INSERT。返回实际写入条数。

        注意：psycopg2 对 float8[] 需要纯 Python list（numpy 数组有 adapter 冲突），写入前统一转换。
        """
        if not items:
            return 0
        conn = self._pool.getconn()
        try:
            existing = self._load_all()
            emat = None
            enorm = None
            if existing:
                emat = np.stack([np.asarray(r["embedding"], dtype=np.float32) for r in existing])
                enorm = np.linalg.norm(emat, axis=1)

            written = 0
            with conn.cursor() as cur:
                for text, vector in items:
                    nvec = np.asarray(vector, dtype=np.float32)
                    if emat is not None:
                        sims = (emat @ nvec) / (enorm * np.linalg.norm(nvec) + 1e-9)
                        if np.max(sims) > DEDUP_THRESHOLD:
                            continue
                    cur.execute(
                        "INSERT INTO memories (text, embedding) VALUES (%s, %s)",
                        (text, list(map(float, vector))),
                    )
                    written += 1
            conn.commit()
            return written
        except Exception:
            conn.rollback()
            logger.warning("[memory_store] PG 批量写入失败", exc_info=True)
            return 0
        finally:
            self._pool.putconn(conn)

    def _touch(self, memory_ids: list[int]) -> None:
        """命中即更新 last_used_at。失败静默。"""
        if self._pool is None or not memory_ids:
            return
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE memories SET last_used_at = now() WHERE id = ANY(%s)",
                    (memory_ids,),
                )
            conn.commit()
        except Exception:
            pass
        finally:
            self._pool.putconn(conn)

    def _prune(self) -> None:
        """行数上限：超限删 created_at 最早的。失败静默。"""
        if self._pool is None:
            return
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM memories")
                total = int(cur.fetchone()[0])
                if total > MAX_MEMORIES:
                    to_delete = total - MAX_MEMORIES
                    cur.execute(
                        "DELETE FROM memories WHERE id IN "
                        "(SELECT id FROM memories ORDER BY created_at ASC LIMIT %s)",
                        (to_delete,),
                    )
            conn.commit()
        except Exception:
            pass
        finally:
            self._pool.putconn(conn)

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
