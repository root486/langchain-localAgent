from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_FAQ_PATH = Path("knowledge") / "E-commerce Data" / "faq.json"
DEFAULT_PREFIX = "请在知识库查询并回答："


@dataclass(frozen=True)
class FaqEntry:
    question: str
    answer: str
    label: str
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline retrieval evaluation for knowledge/E-commerce Data/faq.json."
    )
    parser.add_argument(
        "--faq-path",
        type=Path,
        default=DEFAULT_FAQ_PATH,
        help="Path to the FAQ JSON file, relative to backend/.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Prefix added to every test query to mimic knowledge-skill activation.",
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
    for item in payload:
        if not isinstance(item, dict):
            continue
        entries.append(
            FaqEntry(
                question=str(item.get("question", "")).strip(),
                answer=str(item.get("answer", "")).strip(),
                label=str(item.get("label", "")).strip(),
                url=str(item.get("url", "")).strip(),
            )
        )
    return [entry for entry in entries if entry.question and entry.answer]


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def char_bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}


def overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def score_query(query: str, entry: FaqEntry) -> float:
    normalized_query = normalize(query)
    normalized_question = normalize(entry.question)
    normalized_label = normalize(entry.label)

    if not normalized_query or not normalized_question:
        return 0.0

    score = 0.0

    if normalized_question == normalized_query:
        score += 10.0
    if normalized_question in normalized_query:
        score += 4.0
    if normalized_query in normalized_question:
        score += 2.0

    score += 6.0 * overlap_score(char_bigrams(normalized_query), char_bigrams(normalized_question))

    if normalized_label:
        score += 1.0 * overlap_score(char_bigrams(normalized_query), char_bigrams(normalized_label))

    return score


def rank_entries(query: str, entries: list[FaqEntry]) -> list[tuple[FaqEntry, float]]:
    ranked = [(entry, score_query(query, entry)) for entry in entries]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def softmax_confidences(scores: list[float]) -> list[float]:
    if not scores:
        return []
    anchor = max(scores)
    weights = [math.exp(score - anchor) for score in scores]
    total = sum(weights)
    if not total:
        return [0.0 for _ in scores]
    return [weight / total for weight in weights]


def retrieve(query: str, entries: list[FaqEntry], top_k: int) -> list[FaqEntry]:
    ranked = rank_entries(query, entries)
    return [entry for entry, _ in ranked[:top_k]]


def evaluate(entries: list[FaqEntry], prefix: str, top_k: int) -> dict[str, Any]:
    total = len(entries)
    top1_hits = 0
    topk_hits = 0
    answer_hits = 0
    top1_confidences: list[float] = []
    correct_top1_confidences: list[float] = []
    failures: list[dict[str, Any]] = []

    for index, entry in enumerate(entries, start=1):
        prompt = f"{prefix}{entry.question}"
        ranked = rank_entries(prompt, entries)
        scores = [score for _, score in ranked]
        confidences = softmax_confidences(scores)
        candidates = [candidate for candidate, _ in ranked[:top_k]]
        top1 = ranked[0][0] if ranked else None
        top1_score = ranked[0][1] if ranked else 0.0
        top1_confidence = confidences[0] if confidences else 0.0

        top1_ok = bool(top1 and top1.question == entry.question)
        topk_ok = any(candidate.question == entry.question for candidate in candidates)
        answer_ok = bool(top1 and top1.answer == entry.answer)

        top1_hits += int(top1_ok)
        topk_hits += int(topk_ok)
        answer_hits += int(answer_ok)
        top1_confidences.append(top1_confidence)
        if top1_ok:
            correct_top1_confidences.append(top1_confidence)

        if not top1_ok:
            failures.append(
                {
                    "index": index,
                    "prompt": prompt,
                    "expected_question": entry.question,
                    "expected_label": entry.label,
                    "retrieved_question": top1.question if top1 else "",
                    "retrieved_label": top1.label if top1 else "",
                    "retrieved_url": top1.url if top1 else "",
                    "retrieved_score": round(top1_score, 6),
                    "retrieved_confidence": round(top1_confidence, 6),
                }
            )

    return {
        "total": total,
        "top1_hits": top1_hits,
        "topk_hits": topk_hits,
        "answer_hits": answer_hits,
        "top1_accuracy": top1_hits / total if total else 0.0,
        "topk_accuracy": topk_hits / total if total else 0.0,
        "answer_accuracy": answer_hits / total if total else 0.0,
        "doc_correct_probability": top1_hits / total if total else 0.0,
        "mean_top1_confidence": statistics.mean(top1_confidences) if top1_confidences else 0.0,
        "min_top1_confidence": min(top1_confidences) if top1_confidences else 0.0,
        "max_top1_confidence": max(top1_confidences) if top1_confidences else 0.0,
        "mean_correct_top1_confidence": (
            statistics.mean(correct_top1_confidences) if correct_top1_confidences else 0.0
        ),
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    backend_dir = Path(__file__).resolve().parents[1]
    faq_path = (backend_dir / args.faq_path).resolve()

    entries = load_faq_entries(faq_path)
    result = evaluate(entries, prefix=args.prefix, top_k=max(1, args.top_k))

    print(f"FAQ file: {faq_path}")
    print(f"Total samples: {result['total']}")
    print(f"Query prefix: {args.prefix}")
    print(f"Top-1 retrieval accuracy: {result['top1_accuracy']:.2%} ({result['top1_hits']}/{result['total']})")
    print(f"Top-{max(1, args.top_k)} retrieval accuracy: {result['topk_accuracy']:.2%} ({result['topk_hits']}/{result['total']})")
    print(f"Answer exact-match accuracy: {result['answer_accuracy']:.2%} ({result['answer_hits']}/{result['total']})")
    print(
        "Probability recalled document is correct: "
        f"{result['doc_correct_probability']:.2%} ({result['top1_hits']}/{result['total']})"
    )
    print(
        "Mean top-1 confidence: "
        f"{result['mean_top1_confidence']:.2%} "
        f"(min {result['min_top1_confidence']:.2%}, max {result['max_top1_confidence']:.2%})"
    )

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
