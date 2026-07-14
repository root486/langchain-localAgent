from __future__ import annotations

from typing import AsyncIterator

from knowledge_retrieval.fusion import reciprocal_rank_fusion
from knowledge_retrieval.hybrid_retriever import hybrid_retriever
from knowledge_retrieval.query_expander import expand as expand_queries
from knowledge_retrieval.reranker import rerank_evidences
from knowledge_retrieval.skill_retriever_agent import skill_retriever_agent
from knowledge_retrieval.types import Evidence, OrchestratedRetrievalResult, RetrievalStep, SkillRetrievalResult


class KnowledgeOrchestrator:
    def __init__(self) -> None:
        self.base_dir = None
        self._model_builder = None

    def configure(self, base_dir, model_builder) -> None:
        self.base_dir = base_dir
        self._model_builder = model_builder
        skill_retriever_agent.configure(base_dir, model_builder)
    #把 Skill 检索器返回的原始字典，转成 SkillRetrievalResult 对象。
    def _skill_result_from_payload(self, payload: dict) -> SkillRetrievalResult:
        evidences: list[Evidence] = []
        for item in payload.get("evidences", []):
            if not isinstance(item, dict):
                continue
            score_value = item.get("score")
            try:
                score = float(score_value) if score_value is not None else None
            except (TypeError, ValueError):
                score = None
            raw_parent_id = item.get("parent_id")
            parent_id = str(raw_parent_id).strip() if raw_parent_id else None
            evidences.append(
                Evidence(
                    source_path=str(item.get("source_path", "")),
                    source_type=str(item.get("source_type", "")),
                    locator=str(item.get("locator", "")),
                    snippet=str(item.get("snippet", "")),
                    channel="skill",
                    score=score,
                    parent_id=parent_id,
                )
            )
        return SkillRetrievalResult(
            status=str(payload.get("status", "uncertain")),
            evidences=evidences,
            narrowed_paths=[str(item) for item in payload.get("narrowed_paths", []) if str(item).strip()],
            narrowed_types=[str(item) for item in payload.get("narrowed_types", []) if str(item).strip()],
            rewritten_queries=[str(item) for item in payload.get("rewritten_queries", []) if str(item).strip()],
            searched_paths=[str(item) for item in payload.get("searched_paths", []) if str(item).strip()],
            reason=str(payload.get("reason", "")),
        )

    async def astream(self, query: str) -> AsyncIterator[dict]:
        skill_result: SkillRetrievalResult | None = None

        async for event in skill_retriever_agent.astream(query):
            if event.get("type") == "skill_result":
                skill_result = self._skill_result_from_payload(event["result"])
                continue
            yield event

        if skill_result is None:
            skill_result = SkillRetrievalResult(
                status="uncertain",
                reason="Skill 检索未返回可解析结果。",
            )

        steps: list[RetrievalStep] = [
            RetrievalStep(
                kind="knowledge",
                stage="skill",
                title="Skill 检索结果",
                message=skill_result.reason,
                results=skill_result.evidences[:5],
            )
        ]

        fallback_used = False
        final_evidences = list(skill_result.evidences[:6])
        final_status = skill_result.status
        final_reason = skill_result.reason

        # ---------- Multi-Query ----------
        # 始终执行：将用户口语展开为多个语义等价的检索 query，
        # 每条独立走向量 + BM25，结果汇总后与 Skill 证据一起 RRF 融合。
        # Skill Agent 的 rewritten_queries 作为 BM25 的 query_hints 混用。
        all_queries = await expand_queries(query, self._model_builder)
        all_queries.insert(0, query)  # 原问题也参与检索
        all_vector: list[Evidence] = []
        all_bm25: list[Evidence] = []
        for q in all_queries:
            result = hybrid_retriever.retrieve(
                q,
                top_k=4,
                path_filters=skill_result.narrowed_paths or None,
                query_hints=skill_result.rewritten_queries or None,
            )
            all_vector.extend(result.vector_evidences)
            all_bm25.extend(result.bm25_evidences)

        fallback_used = True
        variants_list = "\n".join(
            f"  {i}. {q}" for i, q in enumerate(all_queries, start=1)
        )
        fallback_message = (
            f"Multi-Query 检索：将「{query}」扩展为：\n{variants_list}\n已汇总向量和 BM25 证据。"
        )
        steps.append(
            RetrievalStep(
                kind="knowledge",
                stage="fallback",
                title="检索策略切换",
                message=fallback_message,
            )
        )

        if all_vector:
            steps.append(
                RetrievalStep(
                    kind="knowledge",
                    stage="vector",
                    title=f"向量检索结果（{len(all_queries)} 路）",
                    message="Multi-Query 向量检索已返回补充证据。",
                    results=all_vector,
                )
            )
        if all_bm25:
            steps.append(
                RetrievalStep(
                    kind="knowledge",
                    stage="bm25",
                    title=f"BM25 检索结果（{len(all_queries)} 路）",
                    message="Multi-Query BM25 检索已返回补充证据。",
                    results=all_bm25,
                )
            )

        # RRF 融合：Skill + 多路 vector + 多路 BM25
        # 取较宽候选池（top_k=20），后续由 qwen3-rerank 精排
        fused_candidates = reciprocal_rank_fusion(
            [
                skill_result.evidences,
                all_vector,
                all_bm25,
            ],
            top_k=20,
        )
        if fused_candidates:
            # qwen3-rerank 精排：从宽池中挑出最相关的 Top-6
            reranked = rerank_evidences(query, fused_candidates, top_n=6)
            final_evidences = reranked
            final_status = "success"
            final_reason = f"RRF 融合 {len(fused_candidates)} 条候选 → qwen3-rerank 精排 → Top {len(reranked)}"
            steps.append(
                RetrievalStep(
                    kind="knowledge",
                    stage="rerank",
                    title="Rerank 精排",
                    message=final_reason,
                    results=reranked,
                )
            )

        yield {
            "type": "orchestrated_result",
            "result": OrchestratedRetrievalResult(
                status=final_status,
                evidences=final_evidences,
                steps=steps,
                fallback_used=fallback_used,
                reason=final_reason,
            ),
        }


knowledge_orchestrator = KnowledgeOrchestrator()
