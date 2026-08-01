#!/usr/bin/env python3
"""FAQ BM25 离线评估：走项目真实 retrieve_bm25 管线（rank_bm25 + jieba 分词）。

用法（从 backend/ 目录运行）：
    python scripts/evaluate_faq_retrieval.py
    python scripts/evaluate_faq_retrieval.py --top-k 3 --show-failures 10

纯离线，不需要 LLM、不需要网络。修改检索逻辑后跑一次，确认 top-1 命中率不退化。
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 允许从 backend/ 或 backend/scripts/ 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.WARNING)  # 屏蔽第三方库（jieba/chromadb）的启动日志

from config import get_settings
from knowledge_retrieval import knowledge_indexer


@dataclass(frozen=True)
class FaqEntry:
    record_id: str  # 与 indexer._split_json 的 record_id 规则一致
    question: str
    answer: str
    label: str
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline BM25 retrieval evaluation on knowledge/E-commerce Data/faq.json."
    )
    parser.add_argument(
        "--faq-path",
        type=Path,
        default=Path("knowledge") / "E-commerce Data" / "faq.json",
        help="Path to the FAQ JSON file, relative to backend/.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many candidates to keep for top-k accuracy.",
    )
    parser.add_argument(
        "--show-failures",
        type=int,
        default=10,
        help="How many failures to print at most.",
    )
    return parser.parse_args()


def load_faq_entries(path: Path) -> list[FaqEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}, got {type(payload).__name__}")

    entries: list[FaqEntry] = []
    # 与 indexer._split_json 相同的 record_id 规则：
    # record_id = item.record_id or item.id or 列表位置(1-based)
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question and not answer:
            continue
        record_id = str(item.get("record_id") or item.get("id") or idx)
        entries.append(
            FaqEntry(
                record_id=record_id,
                question=question,
                answer=answer,
                label=str(item.get("label", "")).strip(),
                url=str(item.get("url", "")).strip(),
            )
        )
    return entries


def _hit(evidence: Any, record_id: str) -> bool:
    """判断检索结果是否命中指定 FAQ 记录（locator 形如 "记录 N" 或 "记录 N (片段 x/y)"）。"""
    return f"记录 {record_id}" in (evidence.locator or "")


def evaluate(entries: list[FaqEntry], top_k: int) -> dict[str, Any]:
    total = 0
    top1_hits = 0
    topk_hits = 0
    top1_scores: list[float] = []
    failures: list[dict[str, Any]] = []

    for entry in entries:
        results = knowledge_indexer.retrieve_bm25(entry.question, top_k=top_k)
        total += 1

        top1_ok = bool(results) and _hit(results[0], entry.record_id)
        topk_ok = any(_hit(r, entry.record_id) for r in results)

        if results and results[0].score is not None:
            top1_scores.append(results[0].score)

        top1_hits += int(top1_ok)
        topk_hits += int(topk_ok)

        if not top1_ok:
            top1 = results[0] if results else None
            failures.append(
                {
                    "expected_question": entry.question,
                    "expected_label": entry.label,
                    "retrieved_locator": getattr(top1, "locator", ""),
                    "retrieved_source": getattr(top1, "source_path", ""),
                    "retrieved_score": round(top1.score, 6) if top1 and top1.score is not None else None,
                    "retrieved_snippet": (getattr(top1, "snippet", "") or "").replace("\n", " ")[:80],
                }
            )

    return {
        "total": total,
        "top1_hits": top1_hits,
        "topk_hits": topk_hits,
        "top1_accuracy": top1_hits / total if total else 0.0,
        "topk_accuracy": topk_hits / total if total else 0.0,
        "mean_top1_score": statistics.mean(top1_scores) if top1_scores else 0.0,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    backend_dir = get_settings().backend_dir
    knowledge_indexer.configure(backend_dir)

    status = knowledge_indexer.status()
    if not status.ready:
        print("[评估] 索引未就绪，正在重建...")
        knowledge_indexer.rebuild_index()
        status = knowledge_indexer.status()
    if not status.bm25_ready:
        print("[评估] 警告：BM25 索引不可用，评估结果为空")

    faq_path = (backend_dir / args.faq_path).resolve()
    entries = load_faq_entries(faq_path)
    result = evaluate(entries, top_k=max(1, args.top_k))

    print(f"FAQ file: {faq_path}")
    print(f"Total samples: {result['total']}")
    print(f"Top-1 retrieval accuracy: {result['top1_accuracy']:.2%} ({result['top1_hits']}/{result['total']})")
    print(f"Top-{max(1, args.top_k)} retrieval accuracy: {result['topk_accuracy']:.2%} ({result['topk_hits']}/{result['total']})")
    print(f"Mean top-1 BM25 score: {result['mean_top1_score']:.3f}")

    failures = result["failures"][: max(0, args.show_failures)]
    if failures:
        print("\nSample failures:")
        for failure in failures:
            print(json.dumps(failure, ensure_ascii=False, indent=2))
    else:
        print("\nNo retrieval failures found in this offline run.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
