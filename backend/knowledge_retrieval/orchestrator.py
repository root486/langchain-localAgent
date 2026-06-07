from __future__ import annotations

from typing import AsyncIterator

from knowledge_retrieval.fusion import reciprocal_rank_fusion
from knowledge_retrieval.hybrid_retriever import hybrid_retriever
from knowledge_retrieval.skill_retriever_agent import skill_retriever_agent
from knowledge_retrieval.types import Evidence, OrchestratedRetrievalResult, RetrievalStep, SkillRetrievalResult


class KnowledgeOrchestrator:
    def __init__(self) -> None:
        self.base_dir = None

    def configure(self, base_dir, model_builder) -> None:
        self.base_dir = base_dir
        skill_retriever_agent.configure(base_dir, model_builder)

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

        should_fallback = (
            skill_result.status in {"partial", "not_found", "uncertain"}
            and (
                not skill_result.narrowed_types
                or any(item in {"md", "json"} for item in skill_result.narrowed_types)
            )
        )

        if should_fallback:
            fallback_used = True
            fallback_message = "Skill 检索未找到充分证据，正在启用向量检索补充结果。"
            steps.append(
                RetrievalStep(
                    kind="knowledge",
                    stage="fallback",
                    title="检索策略切换",
                    message=fallback_message,
                )
            )

            hybrid_result = hybrid_retriever.retrieve(
                query,
                top_k=4,
                path_filters=skill_result.narrowed_paths or None,
                query_hints=skill_result.rewritten_queries or None,
            )

            if hybrid_result.vector_evidences:
                steps.append(
                    RetrievalStep(
                        kind="knowledge",
                        stage="vector",
                        title="向量检索结果",
                        message="向量检索已返回补充证据。",
                        results=hybrid_result.vector_evidences,
                    )
                )
            if hybrid_result.bm25_evidences:
                steps.append(
                    RetrievalStep(
                        kind="knowledge",
                        stage="bm25",
                        title="BM25 检索结果",
                        message="BM25 检索已返回补充证据。",
                        results=hybrid_result.bm25_evidences,
                    )
                )

            fused = reciprocal_rank_fusion(
                [
                    skill_result.evidences,
                    hybrid_result.vector_evidences,
                    hybrid_result.bm25_evidences,
                ],
                top_k=6,
            )
            if fused:
                final_evidences = fused
                final_status = "success"
                final_reason = "已融合 Skill、向量和 BM25 的证据用于回答。"
                steps.append(
                    RetrievalStep(
                        kind="knowledge",
                        stage="fused",
                        title="融合证据",
                        message=final_reason,
                        results=fused,
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
