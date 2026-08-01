from __future__ import annotations

from typing import AsyncIterator

from langsmith import traceable

from knowledge_retrieval.fusion import reciprocal_rank_fusion
from knowledge_retrieval.hybrid_retriever import hybrid_retriever
from knowledge_retrieval.query_expander import expand as expand_queries
from knowledge_retrieval.reranker import rerank_evidences
from knowledge_retrieval.types import Evidence, OrchestratedRetrievalResult, RetrievalStep


class KnowledgeOrchestrator:
    def __init__(self) -> None:
        self.base_dir = None
        self._model_builder = None

    def configure(self, base_dir, model_builder) -> None:
        self.base_dir = base_dir
        self._model_builder = model_builder

    @traceable(run_type="chain", name="knowledge_orchestrator")
    async def astream(self, query: str) -> AsyncIterator[dict]:
        # ---------- Multi-Query ----------
        # 将用户口语展开为多个语义等价的检索 query，每条独立走向量 + BM25，
        # 结果汇总后统一 RRF 融合。此路始终执行。
        all_queries = await expand_queries(query, self._model_builder)
        all_queries.insert(0, query)  # 原问题也参与检索
        all_vector: list[Evidence] = []
        all_bm25: list[Evidence] = []
        for q in all_queries:
            result = hybrid_retriever.retrieve(q, top_k=4)
            all_vector.extend(result.vector_evidences)
            all_bm25.extend(result.bm25_evidences)

        steps: list[RetrievalStep] = []
        if all_vector:
            steps.append(
                RetrievalStep(
                    kind="knowledge",
                    stage="vector",
                    title=f"向量检索结果（{len(all_queries)} 路）",
                    message="Multi-Query 向量检索已返回证据。",
                    results=all_vector,
                )
            )
        if all_bm25:
            steps.append(
                RetrievalStep(
                    kind="knowledge",
                    stage="bm25",
                    title=f"BM25 检索结果（{len(all_queries)} 路）",
                    message="Multi-Query BM25 检索已返回证据。",
                    results=all_bm25,
                )
            )

        if not all_vector and not all_bm25:
            yield {
                "type": "orchestrated_result",
                "result": OrchestratedRetrievalResult(
                    status="not_found",
                    evidences=[],
                    steps=steps,
                    reason="向量与 BM25 均未检索到证据。",
                ),
            }
            return

        # RRF 融合：多路 vector + 多路 BM25
        # 取较宽候选池（top_k=20），后续由 qwen3-rerank 精排
        fused_candidates = reciprocal_rank_fusion(
            [
                all_vector,
                all_bm25,
            ],
            top_k=20,
        )
        final_evidences: list[Evidence] = []
        if fused_candidates:
            # qwen3-rerank 精排：从宽池中挑出最相关的 Top-4
            reranked = rerank_evidences(query, fused_candidates, top_n=4)
            final_evidences = reranked
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
        else:
            final_reason = "RRF 融合未产生候选。"

        yield {
            "type": "orchestrated_result",
            "result": OrchestratedRetrievalResult(
                status="success" if final_evidences else "not_found",
                evidences=final_evidences,
                steps=steps,
                reason=final_reason,
            ),
        }


knowledge_orchestrator = KnowledgeOrchestrator()
