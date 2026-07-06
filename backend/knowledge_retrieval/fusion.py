from __future__ import annotations

from typing import Iterable

from knowledge_retrieval.types import Evidence

#为一条 Evidence 生成唯一的key（文件路径|定位器（文件内位置）|片段前240个字符）。
def _dedupe_key(item: Evidence) -> str:
    normalized_snippet = " ".join(item.snippet.split())
    return f"{item.source_path}|{item.locator}|{normalized_snippet[:240]}"


def reciprocal_rank_fusion(
    evidence_lists: Iterable[list[Evidence]],
    *,
    top_k: int = 6,
    rank_constant: int = 60,
) -> list[Evidence]:
    scores: dict[str, float] = {}
    representatives: dict[str, Evidence] = {}

    for evidence_list in evidence_lists:
        for rank, evidence in enumerate(evidence_list, start=1):
            key = _dedupe_key(evidence)
            scores[key] = scores.get(key, 0.0) + (1.0 / (rank_constant + rank))
            if key not in representatives:# 只存第一个Evidence
                representatives[key] = evidence# 后面遇到同key的就不存了

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)# 把字典变成 (key, value) 元组列表，按分数降序排序
    fused: list[Evidence] = []
    for key, score in ranked[:top_k]:
        evidence = representatives[key]
        fused.append(
            Evidence(
                source_path=evidence.source_path,
                source_type=evidence.source_type,
                locator=evidence.locator,
                snippet=evidence.snippet,
                channel="fused",
                score=score,
                parent_id=evidence.parent_id,
            )
        )
    return fused
