"""长期记忆双端存储：PostgreSQL 结构化事实源 + ChromaDB 向量召回。

职责划分（替代已废弃的 memory_indexer.py 的 MEMORY.md 切块方案）：
- PostgreSQL `memories` 表：长期记忆的权威结构化记录（scope/category/text/status/时间戳/metadata），
  支持精确过滤、UPDATE/DELETE、未来的整合与遗忘（ADD/UPDATE/DELETE 生命周期）。
- ChromaDB `memory_facts` collection：事实文本的 embedding，负责语义召回。
- 写入：PG 插行 + ChromaDB 插条，同一 memory_id 关联；
- 读取：ChromaDB 语义召回 top-k → 按 memory_id 回 PG 取完整结构化事实 → 注入上下文。

PG 或 embedding 任一不可用（未配置 / 连接失败）→ is_ready()=False，检索自动降级（同 redis 降级模式）。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import chromadb

from langsmith import traceable

from config import get_settings
from embeddings_client import EmbeddingClient

logger = logging.getLogger(__name__)

# ChromaDB memory 事实 collection 名（区别于旧 memory_chunks）
MEMORY_COLLECTION_NAME = "memory_facts"
# 相似度阈值：低于该分数视为不相关，不注入上下文（相关性门槛）
MIN_SCORE = 0.4
# 召回放大倍数：ChromaDB 先多召回，过滤阈值后再取 top_k
RETRIEVE_POOL_MULTIPLIER = 3

# 合法枚举值（写入时校验，防止脏数据）
ALLOWED_SCOPES = ("user", "project", "domain", "agent")
ALLOWED_CATEGORIES = ("preference", "fact", "decision", "goal", "constraint")
ALLOWED_STATUSES = ("active", "archived", "superseded")

# ---------- 生命周期硬编码（见 docs/reference/hardcoded-values.md） ----------
# 遗忘：active 记忆超过该天数未使用（或从未使用超过该天数）→ 标记 archived
ARCHIVE_AFTER_DAYS = 180
# 遗忘：记忆行数上限，超限时优先淘汰 archived 中 created_at 最早的
MAX_MEMORIES = 5000
# 抽取器：单次输入的最大消息条数
EXTRACT_MESSAGES_MAX = 20
# 整合决策器：每条新记忆检索的相近已有记忆条数
SIMILAR_TOP_K = 3

# ---------- 生命周期 LLM 提示词（对应 docs/internals/memory-lifecycle-prompts.md） ----------

EXTRACTOR_SYSTEM_PROMPT = (
    "# 角色\n"
    "你是长期记忆抽取器，负责从对话中提炼值得跨会话记住的稳定事实。\n"
    "\n"
    "# 抽取原则\n"
    "1. 只抽「稳定、可复用」的信息：\n"
    "   - 用户偏好（语言、格式、工作方式、工具偏好）\n"
    "   - 个人/团队属性（职业、行业、技能、角色）\n"
    "   - 项目/领域事实（项目背景、技术选型、业务规则）\n"
    "   - 决策与目标（明确了的选择、正在追求的目标）\n"
    "   - 长期约束（红线、必须遵守的规则）\n"
    "2. 不抽：寒暄、一次性请求、纯提问、情绪、偶然的临时信息。\n"
    "3. 一条记忆 = 一个原子陈述（单一事实），用陈述句、第三人称，不含多个要点。\n"
    "4. 与已有记忆重复的内容不重复抽取；用户最新表述与旧信息冲突时，以最新为准。\n"
    "5. 对话中没有值得记住的内容 → 返回空数组。\n"
    "\n"
    "# 输出（严格 JSON，不要任何多余文字）\n"
    '{\n'
    '  "memories": [\n'
    '    {"scope": "user", "category": "preference", "text": "User 偏好英文交流", "confidence": 0.9}\n'
    '  ]\n'
    '}\n'
    "- scope ∈ user | project | domain | agent\n"
    "- category ∈ preference | fact | decision | goal | constraint\n"
    "- confidence: 0~1，抽取器对「该信息确实稳定可复用」的确信度\n"
)

CONSOLIDATOR_SYSTEM_PROMPT = (
    "# 角色\n"
    "你是记忆整合决策器。系统会给出若干条「新抽取的记忆」，每条附带与之语义相近的「已有记忆」。\n"
    "请对每条新记忆决定一个操作：ADD / UPDATE / DELETE / NOOP。\n"
    "\n"
    "# 操作定义\n"
    "- ADD    : 新记忆与任何已有记忆都无实质重叠 → 新增一条。\n"
    "- UPDATE : 某条已有记忆主题相同但内容不同/过时，新记忆应取代它 → 修改该记忆（合并成信息更完整的版本）。\n"
    "- DELETE : 新记忆与某条已有记忆直接矛盾并推翻它 → 将该记忆标记为 superseded（不再被检索）。\n"
    "- NOOP   : 已有记忆已完整覆盖该信息 → 不做任何操作。\n"
    "\n"
    "# 规则\n"
    "1. 判断依据是「语义等价」，不是字面相同。\n"
    "2. 冲突时以新记忆为准（用户最新表达优先于旧记忆）。\n"
    "3. UPDATE / DELETE 必须给出 target_id（指向已有记忆的 id）。\n"
    "4. UPDATE 必须给出 new_text（合并后信息更完整的版本）。\n"
    "5. 每个 candidate_index 恰好输出一条 operation。\n"
    "6. 输出必须严格 JSON，不要任何多余文字。\n"
    "\n"
    "# 输出格式\n"
    '{\n'
    '  "operations": [\n'
    '    {"candidate_index": 0, "operation": "ADD", "target_id": null, "new_text": null, "reason": "全新信息，无冲突"}\n'
    '  ]\n'
    '}\n'
)


class MemoryStore:
    def __init__(self) -> None:
        self.base_dir: Path | None = None
        self._pool: Any = None  # psycopg2 SimpleConnectionPool
        self._chroma_client: Any | None = None  # ChromaDB PersistentClient（懒加载）
        self._collection: Any | None = None  # ChromaDB memory_facts collection
        self._embedding_ok = False

    # ---------- 生命周期 ----------

    def configure(self, base_dir: Path) -> None:
        """初始化 PG 连接池 + ChromaDB collection。任一失败 → 降级（is_ready()=False）。"""
        self.base_dir = base_dir
        self._embedding_ok = bool(get_settings().embedding_api_key)

        # ChromaDB 侧（懒加载，不阻塞启动）
        if self._embedding_ok:
            try:
                collection = self._get_collection()
                self._collection = collection
            except Exception:
                logger.warning("[memory_store] ChromaDB 初始化失败，向量检索降级", exc_info=True)
                self._collection = None

        # PG 侧
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
        """PG 与向量两端都可用才返回 True。"""
        return self.base_dir is not None and self._pool is not None and self._collection is not None

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
        return {
            "ready": self.is_ready(),
            "pg": self._pool is not None,
            "vector": self._collection is not None,
            "count": count,
        }

    # ---------- 写入（本轮用于种子数据，为下一轮自动抽取/整合打底） ----------

    def add(
        self,
        text: str,
        scope: str = "user",
        category: str = "fact",
        metadata: dict[str, Any] | None = None,
    ) -> int | None:
        """写入一条长期记忆：PG 插行返回 memory_id，ChromaDB 写入同 id 文档。失败返回 None。"""
        if not self.is_ready() or not text.strip():
            return None
        if scope not in ALLOWED_SCOPES:
            raise ValueError(f"scope 必须是 {ALLOWED_SCOPES} 之一，收到 {scope!r}")
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"category 必须是 {ALLOWED_CATEGORIES} 之一，收到 {category!r}")

        memory_id = self._insert_pg(text, scope, category, metadata or {})
        if memory_id is None:
            return None
        ok = self._upsert_vector(memory_id, text, scope, category)
        if not ok:
            # 向量写入失败 → 回滚 PG 行，保持双端一致
            self.delete(memory_id)
            return None
        return memory_id

    def delete(self, memory_id: int) -> bool:
        """删除一条记忆（PG 行 + ChromaDB 文档），保持双端一致。"""
        if self._pool is None:
            return False
        try:
            conn = self._pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
                conn.commit()
            finally:
                self._pool.putconn(conn)
        except Exception:
            return False
        if self._collection is not None:
            try:
                self._collection.delete(ids=[str(memory_id)])
            except Exception:
                pass
        return True

    def update_text(self, memory_id: int, new_text: str) -> bool:
        """UPDATE 操作：把某条记忆替换为更完整的合并版本（PG 行 text + 重新嵌入覆盖向量文档）。

        保留原有 scope / category / metadata，只更新 text。
        """
        if not self.is_ready() or not str(new_text or "").strip():
            return False
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE memories SET text = %s, updated_at = now() "
                    "WHERE id = %s AND status = 'active'",
                    (new_text, memory_id),
                )
                updated = cur.rowcount > 0
            conn.commit()
        except Exception:
            conn.rollback()
            logger.warning("[memory_store] UPDATE PG 失败", exc_info=True)
            return False
        finally:
            self._pool.putconn(conn)
        if not updated:
            return False
        # 取原 scope/category 用于向量覆盖
        row = self._fetch_by_ids([memory_id]).get(memory_id)
        scope = row["scope"] if row else "user"
        category = row["category"] if row else "fact"
        if self._collection is not None:
            self._upsert_vector(memory_id, new_text, scope, category)
        return True

    def supersede(self, memory_id: int) -> bool:
        """DELETE 操作：标记为 superseded（不再被检索），并移除 ChromaDB 文档。

        改口推翻旧记忆时使用；向量端删除，避免整合决策器再次把它当「相近已有」。
        """
        if self._pool is None:
            return False
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE memories SET status = 'superseded', updated_at = now() "
                    "WHERE id = %s AND status = 'active'",
                    (memory_id,),
                )
                ok = cur.rowcount > 0
            conn.commit()
        except Exception:
            conn.rollback()
            logger.warning("[memory_store] supersede 失败", exc_info=True)
            return False
        finally:
            self._pool.putconn(conn)
        if ok and self._collection is not None:
            self._delete_vector([memory_id])
        return ok

    # ---------- 读取 ----------

    @traceable(
        run_type="retriever",
        name="memory_retrieve",
        process_outputs=lambda o: {
            "n": len(o),
            "memory_ids": [item.get("memory_id") for item in o],
        },
    )
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        """语义召回：ChromaDB 召回 → 回 PG 取完整事实 → 阈值过滤 + 排序。

        返回 `[{memory_id, text, score, scope, category, source, metadata}]`，
        与旧 `memory_indexer.retrieve` 的 `{text, score, source}` 字段兼容。
        """
        if not self.is_ready() or not query.strip():
            return []

        try:
            query_embedding = EmbeddingClient().embed_one(query)
            n_results = max(1, top_k * RETRIEVE_POOL_MULTIPLIER)
            where = {"scope": scope} if scope else None
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["distances"],  # ids 始终随结果返回，无需显式 include
            )
        except Exception:
            logger.warning("[memory_store] 向量召回失败", exc_info=True)
            return []

        ids_str = results.get("ids", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []
        if not ids_str:
            return []

        try:
            memory_ids = [int(i) for i in ids_str]
        except ValueError:
            return []

        rows = self._fetch_by_ids(memory_ids)
        if not rows:
            return []

        # 合并分数
        scored: list[dict[str, Any]] = []
        for i, mem_id in enumerate(memory_ids):
            if mem_id not in rows:
                continue
            distance = float(distances[i]) if i < len(distances) else 1.0
            score = 1.0 - distance
            if score < MIN_SCORE:
                continue
            row = rows[mem_id]
            scored.append(
                {
                    "memory_id": mem_id,
                    "text": row["text"],
                    "score": round(score, 4),
                    "scope": row["scope"],
                    "category": row["category"],
                    "source": f"memory:{row['scope']}/{row['category']}",
                    "metadata": row["metadata"],
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        scored = scored[:top_k]
        # 命中即更新 last_used_at（遗忘策略依据，见 run_forget_rules）
        if scored:
            self._touch([item["memory_id"] for item in scored])
        return scored

    # ---------- 生命周期：自动写入（抽取器） + 整合（决策器） + 遗忘（规则） ----------
    # 对应 docs/internals/memory-lifecycle-prompts.md 第一~三节

    async def write_from_session(self, messages: list[dict[str, Any]]) -> int:
        """从会话消息自动写入长期记忆：抽取器 → 整合决策器 → 执行。

        返回实际写入/更新的条数（NOOP、解析失败、LLM 异常均不计入）。
        任何一步失败都不抛出，只打日志（调用方是后台任务，不能影响 SSE 响应）。
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

        memory_summary = self._build_memory_summary()
        facts = await self._extract_facts(recent, memory_summary)
        if not facts:
            return 0

        # 为每条候选检索相近的 active 已有记忆，供决策器判断
        candidates: list[dict[str, Any]] = []
        for fact in facts:
            candidates.append({"fact": fact, "similar": self._find_similar(fact["text"])})

        operations = await self._decide_operations(candidates)
        written = 0
        for cand, op in zip(candidates, operations):
            written += self._apply_operation(cand["fact"], op)
        return written

    async def _extract_facts(
        self,
        messages: list[dict[str, str]],
        memory_summary: str,
    ) -> list[dict[str, Any]]:
        """抽取器：对话 → 原子事实。解析失败 / 空数组 → []。"""
        prompt = (
            EXTRACTOR_SYSTEM_PROMPT
            + "\n\n# 对话记录\n"
            + json.dumps(messages, ensure_ascii=False)
            + "\n\n# 已有记忆摘要（可选，帮助避免重复抽取）\n"
            + (memory_summary or "（无）")
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

        facts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in data.get("memories") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue  # 同一批内去重
            seen.add(key)
            scope = str(item.get("scope", "") or "")
            category = str(item.get("category", "") or "")
            if scope not in ALLOWED_SCOPES:
                scope = "user"
            if category not in ALLOWED_CATEGORIES:
                category = "fact"
            confidence = item.get("confidence")
            facts.append(
                {
                    "text": text,
                    "scope": scope,
                    "category": category,
                    "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
                }
            )
        return facts

    async def _decide_operations(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """整合决策器：每条候选 + 相近已有记忆 → ADD/UPDATE/DELETE/NOOP。

        返回与 candidates 等长的列表；非法/缺省 → ADD 兜底（对应文档调用约定）。
        """
        lines = ["# 待处理的新记忆（每条附带相近已有记忆）"]
        for idx, cand in enumerate(candidates):
            fact = cand["fact"]
            lines.append(
                f'{idx + 1}. 新记忆: "{fact["text"]}" (scope={fact["scope"]}, category={fact["category"]})'
            )
            similar = cand.get("similar") or []
            if not similar:
                lines.append("   相近已有: (无)")
            else:
                lines.append("   相近已有:")
                for mem in similar:
                    lines.append(
                        f"     - id={mem['id']}: \"{mem['text']}\" (scope={mem.get('scope')}, category={mem.get('category')})"
                    )
        prompt = CONSOLIDATOR_SYSTEM_PROMPT + "\n\n" + "\n".join(lines)

        fallback = [{"operation": "ADD", "target_id": None, "new_text": None} for _ in candidates]
        try:
            response = await self._build_llm().ainvoke(
                [{"role": "system", "content": prompt}]
            )
        except Exception:
            logger.warning("[memory_store] 记忆整合 LLM 调用失败，全部按 ADD 兜底", exc_info=True)
            return fallback
        data = self._parse_json_block(_llm_text(getattr(response, "content", "")))
        if not data:
            return fallback

        by_index: dict[int, dict[str, Any]] = {}
        for op in data.get("operations") or []:
            if not isinstance(op, dict):
                continue
            try:
                idx = int(op.get("candidate_index", -1))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(candidates):
                by_index[idx] = op

        operations: list[dict[str, Any]] = []
        for idx in range(len(candidates)):
            op = by_index.get(idx, {})
            operation = str(op.get("operation", "") or "").upper()
            if operation not in ("ADD", "UPDATE", "DELETE", "NOOP"):
                operation = "ADD"
            new_text = str(op.get("new_text") or "").strip() or None
            # UPDATE 必须有 target_id + new_text；DELETE 必须有 target_id，否则退化 ADD
            if operation == "UPDATE" and (op.get("target_id") is None or not new_text):
                operation = "ADD"
            if operation == "DELETE" and op.get("target_id") is None:
                operation = "ADD"
            operations.append(
                {"operation": operation, "target_id": op.get("target_id"), "new_text": new_text}
            )
        return operations

    def _apply_operation(self, fact: dict[str, Any], op: dict[str, Any]) -> int:
        """执行一条决策操作，返回实际写入/改动的条数（NOOP 或失败返回 0）。"""
        operation = op["operation"]
        try:
            if operation == "ADD":
                mid = self.add(
                    fact["text"],
                    scope=fact["scope"],
                    category=fact["category"],
                    metadata={"source": "auto_extract", "confidence": fact.get("confidence")},
                )
                return 1 if mid is not None else 0
            if operation == "UPDATE":
                target_id = _coerce_id(op.get("target_id"))
                new_text = (op.get("new_text") or "").strip()
                if target_id is None or not new_text:
                    return 0
                return 1 if self.update_text(target_id, new_text) else 0
            if operation == "DELETE":
                target_id = _coerce_id(op.get("target_id"))
                if target_id is None:
                    return 0
                return 1 if self.supersede(target_id) else 0
        except Exception:
            logger.warning("[memory_store] 记忆操作执行失败: %s", operation, exc_info=True)
            return 0
        return 0  # NOOP

    def _find_similar(self, text: str, top_k: int = SIMILAR_TOP_K) -> list[dict[str, Any]]:
        """整合决策器用：ChromaDB 语义召回相近的 active 已有记忆（不设 MIN_SCORE 门槛）。"""
        if self._collection is None or not text.strip():
            return []
        try:
            embedding = EmbeddingClient().embed_one(text)
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["distances"],
            )
        except Exception:
            logger.warning("[memory_store] 相近记忆召回失败", exc_info=True)
            return []
        memory_ids: list[int] = []
        for raw in results.get("ids", [[]])[0] or []:
            try:
                memory_ids.append(int(raw))
            except (TypeError, ValueError):
                continue
        if not memory_ids:
            return []
        rows = self._fetch_by_ids(memory_ids)  # 只返回 status='active'
        similar: list[dict[str, Any]] = []
        for mid in memory_ids:
            row = rows.get(mid)
            if row is None:
                continue
            similar.append(
                {"id": mid, "text": row["text"], "scope": row["scope"], "category": row["category"]}
            )
        return similar

    def _build_memory_summary(self, limit: int = 8) -> str:
        """已有记忆摘要（帮助抽取器避免重复抽取）。无则返回空串。"""
        recent = self._recent_memories(limit)
        if not recent:
            return ""
        return "\n".join(f"- (id={item['id']}) {item['text']}" for item in recent)

    def _recent_memories(self, limit: int = 8) -> list[dict[str, Any]]:
        """最近更新的 active 记忆（按 updated_at 倒序）。"""
        if self._pool is None:
            return []
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, text FROM memories WHERE status = 'active' "
                    "ORDER BY updated_at DESC LIMIT %s",
                    (limit,),
                )
                return [{"id": int(mid), "text": text} for mid, text in cur.fetchall()]
        except Exception:
            logger.warning("[memory_store] 读取近期记忆失败", exc_info=True)
            return []
        finally:
            self._pool.putconn(conn)

    def run_forget_rules(self) -> dict[str, int]:
        """遗忘策略（纯规则，非 LLM，见文档第三节）。每次启动执行一次。

        1) active 记忆超过 ARCHIVE_AFTER_DAYS 未使用（从未使用则按 created_at 计）→ archived；
        2) 行数上限 MAX_MEMORIES → 淘汰 archived 中 created_at 最早的。
        返回 {"archived": n, "pruned": n}。
        """
        result: dict[str, int] = {"archived": 0, "pruned": 0}
        if self._pool is None:
            return result

        conn = self._pool.getconn()
        try:
            # 1) 归档长期未用的记忆。注意：刚写入的新记忆 last_used_at 为空，不应立即归档，
            #    因此从未使用的记忆按 created_at 计满阈值才归档。
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE memories SET status = 'archived', updated_at = now()
                    WHERE status = 'active'
                      AND (
                        (last_used_at IS NOT NULL AND last_used_at < now() - make_interval(days => %s))
                        OR (last_used_at IS NULL AND created_at < now() - make_interval(days => %s))
                      )
                    """,
                    (ARCHIVE_AFTER_DAYS, ARCHIVE_AFTER_DAYS),
                )
                result["archived"] = cur.rowcount or 0
            conn.commit()

            # 2) 行数上限：优先删除 archived 中 created_at 最早的
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM memories")
                total = int(cur.fetchone()[0])
            if total > MAX_MEMORIES:
                to_delete = total - MAX_MEMORIES
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM memories WHERE status = 'archived' "
                        "ORDER BY created_at ASC LIMIT %s",
                        (to_delete,),
                    )
                    prune_ids = [int(r[0]) for r in cur.fetchall()]
                if prune_ids:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM memories WHERE id = ANY(%s)", (prune_ids,))
                    conn.commit()
                    result["pruned"] = len(prune_ids)
                    self._delete_vector(prune_ids)
        except Exception:
            conn.rollback()
            logger.warning("[memory_store] 遗忘规则执行失败", exc_info=True)
        finally:
            self._pool.putconn(conn)
        return result

    def _touch(self, memory_ids: list[int]) -> None:
        """命中即更新 last_used_at（遗忘策略依据）。失败静默。"""
        if self._pool is None or not memory_ids:
            return
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE memories SET last_used_at = now() "
                    "WHERE id = ANY(%s) AND status = 'active'",
                    (memory_ids,),
                )
            conn.commit()
        except Exception:
            pass
        finally:
            self._pool.putconn(conn)

    def _delete_vector(self, memory_ids: list[int]) -> None:
        """按 id 批量删除 ChromaDB 文档。失败静默。"""
        if self._collection is None or not memory_ids:
            return
        try:
            self._collection.delete(ids=[str(i) for i in memory_ids])
        except Exception:
            pass

    def _build_llm(self) -> Any:
        """记忆生命周期专用轻量 LLM：优先 SUMMARY_MODEL（便宜），未配置回退主模型。

        与 agent_manager._build_summary_model 语义一致，但内聚在本模块，
        避免与 graph/agent.py 循环依赖。
        """
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
        """从 LLM 输出中稳健提取 JSON 对象（容忍 markdown 围栏 / 前后废话）。

        先整体解析；失败则取第一个 `{...}` 片段再解析。仍失败返回 None。
        """
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

    # ---------- 内部实现 ----------

    def _get_collection(self) -> Any:
        if self._chroma_client is None:
            from chromadb.config import Settings as ChromaSettings

            self._chroma_client = chromadb.PersistentClient(
                path=str(self.base_dir / "storage" / "memory_facts" / "chroma"),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._chroma_client.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _ensure_schema(self) -> None:
        if self._pool is None:
            return
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id           BIGSERIAL PRIMARY KEY,
                        scope        TEXT NOT NULL DEFAULT 'user',
                        category     TEXT NOT NULL DEFAULT 'fact',
                        text         TEXT NOT NULL,
                        metadata     JSONB NOT NULL DEFAULT '{}',
                        status       TEXT NOT NULL DEFAULT 'active',
                        created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                        last_used_at TIMESTAMPTZ
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories (scope, category)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_status ON memories (status)"
                )
            conn.commit()
        finally:
            self._pool.putconn(conn)

    def _insert_pg(
        self,
        text: str,
        scope: str,
        category: str,
        metadata: dict[str, Any],
    ) -> int | None:
        if self._pool is None:
            return None
        from psycopg2.extras import Json

        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memories (scope, category, text, metadata, status)
                    VALUES (%s, %s, %s, %s, 'active')
                    RETURNING id
                    """,
                    (scope, category, text, Json(metadata)),
                )
                memory_id = int(cur.fetchone()[0])
            conn.commit()
            return memory_id
        except Exception:
            conn.rollback()
            logger.warning("[memory_store] PG 插入失败", exc_info=True)
            return None
        finally:
            self._pool.putconn(conn)

    def _upsert_vector(self, memory_id: int, text: str, scope: str, category: str) -> bool:
        if self._collection is None:
            return False
        try:
            embedding = EmbeddingClient().embed_one(text)
            self._collection.upsert(
                ids=[str(memory_id)],
                documents=[text],
                embeddings=[embedding],
                metadatas=[{"memory_id": memory_id, "scope": scope, "category": category}],
            )
            return True
        except Exception:
            logger.warning("[memory_store] ChromaDB 写入失败", exc_info=True)
            return False

    def _fetch_by_ids(self, memory_ids: list[int]) -> dict[int, dict[str, Any]]:
        """按 id 批量回 PG 取结构化事实，返回 {id: {text, scope, category, metadata}}。"""
        if self._pool is None or not memory_ids:
            return {}
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, scope, category, text, metadata
                    FROM memories
                    WHERE id = ANY(%s) AND status = 'active'
                    """,
                    (memory_ids,),
                )
                rows: dict[int, dict[str, Any]] = {}
                for mem_id, scope, category, text, metadata in cur.fetchall():
                    rows[int(mem_id)] = {
                        "text": text,
                        "scope": scope,
                        "category": category,
                        "metadata": metadata if isinstance(metadata, dict) else {},
                    }
                return rows
        except Exception:
            logger.warning("[memory_store] PG 批量读取失败", exc_info=True)
            return {}
        finally:
            self._pool.putconn(conn)


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


def _coerce_id(value: Any) -> int | None:
    """把 target_id（可能是 int / str / None）安全转成 int。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


memory_store = MemoryStore()
