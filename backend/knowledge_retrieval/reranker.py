"""Reranker: 调用百炼 qwen3-rerank 对检索候选做精排。

在 RRF 融合产生较宽候选池（~20 条）后，用交叉编码器精排到 Top-K，
提升 Precision 并降低 LLM 上下文中的噪音。
"""

from __future__ import annotations

from langsmith import traceable

from knowledge_retrieval.types import Evidence

RERANK_MODEL = "qwen3-rerank"
RERANK_API_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)


@traceable(
    run_type="chain",
    name="rerank",
    metadata={"model": RERANK_MODEL},
)
def rerank_evidences(
    query: str,
    evidences: list[Evidence],
    *,
    top_n: int = 4,
    model: str = RERANK_MODEL,
) -> list[Evidence]:
    """用百炼 qwen3-rerank 对 Evidence 列表重排序，返回得分最高的 top_n 条。

    失败时自动降级：返回原始顺序的前 top_n 条，不影响管线可用性。
    """
    if not evidences:
        return []
    if len(evidences) <= top_n:
        return evidences

    from config import get_settings

    settings = get_settings()
    api_key = settings.embedding_api_key  # DASHSCOPE_API_KEY
    if not api_key:
        print("[Reranker] 未配置 DASHSCOPE_API_KEY，跳过重排序")
        return evidences[:top_n]

    documents: list[str] = [ev.snippet for ev in evidences]

    import json
    import urllib.request

    try:
        req = urllib.request.Request(
            RERANK_API_URL,
            data=json.dumps(
                {
                    "model": model,
                    "input": {
                        "query": query,
                        "documents": documents,
                    },
                    "parameters": {"top_n": top_n},
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:  # type: ignore[attr-defined]
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"[Reranker] 重排序失败，降级为原始排序: {exc}")
        return evidences[:top_n]

    results: list[dict] = data.get("output", {}).get("results", [])
    if not results:
        return evidences[:top_n]

    # 按 relevance_score 降序排列
    sorted_results = sorted(
        results, key=lambda r: r.get("relevance_score", 0), reverse=True
    )

    reranked: list[Evidence] = []
    for item in sorted_results:
        idx = int(item.get("index", 0))
        if 0 <= idx < len(evidences):
            ev = evidences[idx]
            reranked.append(
                Evidence(
                    source_path=ev.source_path,
                    source_type=ev.source_type,
                    locator=ev.locator,
                    snippet=ev.snippet,
                    channel="fused",  # rerank 只重排序，不改变来源 channel
                    score=item.get("relevance_score", ev.score),
                    parent_id=ev.parent_id,
                )
            )
        if len(reranked) >= top_n:
            break

    return reranked or evidences[:top_n]
