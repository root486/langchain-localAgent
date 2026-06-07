from __future__ import annotations

from knowledge_retrieval.indexer import knowledge_indexer
from knowledge_retrieval.types import HybridRetrievalResult


class HybridRetriever:
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 4,
        path_filters: list[str] | None = None,
        query_hints: list[str] | None = None,
    ) -> HybridRetrievalResult:
        return HybridRetrievalResult(
            vector_evidences=knowledge_indexer.retrieve_vector(
                query,
                top_k=top_k,
                path_filters=path_filters,
            ),
            bm25_evidences=knowledge_indexer.retrieve_bm25(
                query,
                top_k=top_k,
                path_filters=path_filters,
                query_hints=query_hints,
            ),
        )


hybrid_retriever = HybridRetriever()
