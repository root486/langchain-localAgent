from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_FAQ_PATH = Path(r"E:\download\ragclaw\ragclaw\backend\knowledge\E-commerce Data\faq.json")
DEFAULT_PREFIX = "请帮我在知识库查询并回答："
DEFAULT_OUTPUT_PATH = Path("storage") / "eval_outputs" / "faq_system_accuracy_results.json"


@dataclass(frozen=True)
class FaqEntry:
    question: str
    answer: str
    label: str
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the current knowledge-base system against a FAQ JSON file."
    )
    parser.add_argument(
        "--faq-path",
        type=Path,
        default=DEFAULT_FAQ_PATH,
        help="Absolute or relative path to the FAQ JSON file.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Prefix used to trigger the knowledge-base workflow.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N FAQ entries. 0 means all entries.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Relative output path under backend/ for the raw result JSON.",
    )
    parser.add_argument(
        "--show-failures",
        type=int,
        default=10,
        help="How many failed samples to print.",
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
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        entries.append(
            FaqEntry(
                question=question,
                answer=answer,
                label=str(item.get("label", "")).strip(),
                url=str(item.get("url", "")).strip(),
            )
        )
    return entries


def normalize(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\s+", "", lowered)
    lowered = re.sub(r"[^\w\u4e00-\u9fff]+", "", lowered)
    return lowered


def answer_similarity(left: str, right: str) -> float:
    normalized_left = normalize(left)
    normalized_right = normalize(right)
    if not normalized_left or not normalized_right:
        return 0.0
    return SequenceMatcher(a=normalized_left, b=normalized_right).ratio()


def exact_match(left: str, right: str) -> bool:
    return normalize(left) == normalize(right)


def clean_answer_for_eval(answer: str) -> str:
    text = answer.replace("\r\n", "\n").strip()
    if not text:
        return ""

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^#{1,6}\s*", line):
            continue
        if line.startswith("来源") or line.startswith("文件路径") or line.startswith("Source:"):
            continue
        if re.match(r"^(https?://|www\.)", line, re.IGNORECASE):
            continue

        line = re.sub(r"`([^`]*)`", r"\1", line)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"^\s*[-*]\s+", "", line)
        line = re.sub(r"^\s*\d+\.\s+", "", line)
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or text


def build_contexts_from_results(results: list[dict[str, Any]]) -> list[str]:
    contexts: list[str] = []
    seen: set[str] = set()

    for item in results:
        source_path = str(item.get("source_path", "")).strip()
        locator = str(item.get("locator", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        header_parts = [part for part in [source_path, locator] if part]
        header = " | ".join(header_parts)
        content = f"{header}\n{snippet}".strip() if header else snippet
        if not content or content in seen:
            continue
        seen.add(content)
        contexts.append(content)

    return contexts


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def initialize_backend(backend_dir: Path):
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    from graph.agent import agent_manager
    from graph.memory_indexer import memory_indexer
    from knowledge_retrieval import knowledge_indexer
    from tools.skills_scanner import refresh_snapshot

    refresh_snapshot(backend_dir)
    agent_manager.initialize(backend_dir)
    memory_indexer.configure(backend_dir)
    memory_indexer.rebuild_index()
    knowledge_indexer.configure(backend_dir)
    knowledge_indexer.rebuild_index()
    return agent_manager


async def run_single_question(question: str, prefix: str) -> tuple[str, list[str]]:
    from graph.agent import agent_manager

    final_answer = ""
    streamed_parts: list[str] = []
    final_knowledge_results: list[dict[str, Any]] = []

    async for event in agent_manager.astream(f"{prefix}{question}", history=[]):
        event_type = str(event.get("type", ""))
        if event_type == "retrieval":
            if event.get("kind") == "knowledge" and event.get("results"):
                final_knowledge_results = [
                    item for item in event.get("results", []) if isinstance(item, dict)
                ]
        elif event_type == "token":
            streamed_parts.append(str(event.get("content", "")))
        elif event_type == "done":
            final_answer = str(event.get("content", "")).strip()

    answer = final_answer or "".join(streamed_parts).strip()
    contexts = build_contexts_from_results(final_knowledge_results)
    return answer, contexts


def build_ragas_dataset_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "user_input": row["user_input"],
            "response": row["response"],
            "retrieved_contexts": row["retrieved_contexts"],
            "reference": row["reference"],
        }
        for row in rows
    ]


def run_ragas(rows: list[dict[str, Any]]) -> tuple[Any | None, dict[str, Any] | None]:
    try:
        from ragas import EvaluationDataset, evaluate as ragas_evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            NonLLMContextPrecisionWithReference,
            NonLLMContextRecall,
            ResponseRelevancy,
        )
        from langchain_openai import OpenAIEmbeddings

        from config import get_settings
        from graph.agent import agent_manager
    except ImportError as exc:
        return None, {"enabled": False, "error": f"Missing evaluation dependency: {exc}"}

    samples = [
        SingleTurnSample(
            user_input=row["user_input"],
            response=row["response"],
            retrieved_contexts=row["retrieved_contexts"],
            reference=row["reference"],
        )
        for row in rows
    ]
    dataset = EvaluationDataset(samples=samples)

    ragas_llm = LangchainLLMWrapper(agent_manager._build_chat_model())
    settings = get_settings()
    ragas_embeddings = None
    if settings.embedding_api_key:
        ragas_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_base_url,
            )
        )

    metrics: list[Any] = [
        NonLLMContextPrecisionWithReference(),
        NonLLMContextRecall(),
        Faithfulness(),
    ]
    if ragas_embeddings is not None:
        metrics.append(ResponseRelevancy())

    result = ragas_evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
        show_progress=True,
    )
    summary = {
        "enabled": True,
        "metrics": [metric.name for metric in metrics],
        "scores": {
            key: value.item() if hasattr(value, "item") else value
            for key, value in getattr(result, "_repr_dict", {}).items()
        },
    }
    return result.to_pandas(), summary


async def evaluate_system(args: argparse.Namespace) -> dict[str, Any]:
    backend_dir = Path(__file__).resolve().parents[1]
    faq_path = args.faq_path if args.faq_path.is_absolute() else (backend_dir / args.faq_path).resolve()
    if not faq_path.exists():
        raise FileNotFoundError(f"FAQ file not found: {faq_path}")

    initialize_backend(backend_dir)
    entries = load_faq_entries(faq_path)
    if args.limit > 0:
        entries = entries[: args.limit]

    rows: list[dict[str, Any]] = []
    exact_hits = 0
    similarity_scores: list[float] = []

    for index, entry in enumerate(entries, start=1):
        answer, contexts = await run_single_question(entry.question, args.prefix)
        cleaned_answer = clean_answer_for_eval(answer)
        similarity = answer_similarity(cleaned_answer, entry.answer)
        is_exact = exact_match(cleaned_answer, entry.answer)
        exact_hits += int(is_exact)
        similarity_scores.append(similarity)
        rows.append(
            {
                "index": index,
                "question": entry.question,
                "user_input": entry.question,
                "response": cleaned_answer,
                "raw_response": answer,
                "retrieved_contexts": contexts,
                "reference": entry.answer,
                "exact_match": is_exact,
                "answer_similarity": similarity,
                "label": entry.label,
                "url": entry.url,
            }
        )

    ragas_df, ragas_summary = run_ragas(build_ragas_dataset_payload(rows))
    total = len(rows)
    mean_similarity = sum(similarity_scores) / total if total else 0.0

    return {
        "faq_path": str(faq_path),
        "total": total,
        "exact_match_accuracy": exact_hits / total if total else 0.0,
        "mean_answer_similarity": mean_similarity,
        "rows": rows,
        "ragas_summary": ragas_summary,
        "ragas_dataframe": ragas_df.to_dict(orient="records") if ragas_df is not None else None,
    }


def write_output(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(sanitize_json(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_summary(summary: dict[str, Any], show_failures: int) -> None:
    print(f"FAQ file: {summary['faq_path']}")
    print(f"Total samples: {summary['total']}")
    print(f"Exact-match accuracy: {summary['exact_match_accuracy']:.2%}")
    print(f"Mean answer similarity: {summary['mean_answer_similarity']:.2%}")

    ragas_summary = summary.get("ragas_summary")
    if isinstance(ragas_summary, dict):
        print("\nRagas summary:")
        if ragas_summary.get("enabled"):
            for metric_name, value in ragas_summary.get("scores", {}).items():
                if isinstance(value, (int, float)):
                    print(f"- {metric_name}: {value:.4f}")
                else:
                    print(f"- {metric_name}: {value}")
        else:
            print(f"- skipped: {ragas_summary.get('error', 'unknown reason')}")

    failures = [
        {
            "index": row["index"],
            "question": row["question"],
            "response": row["raw_response"][:300],
            "reference": row["reference"][:300],
            "answer_similarity": round(float(row["answer_similarity"]), 4),
            "retrieved_context_count": len(row["retrieved_contexts"]),
        }
        for row in summary["rows"]
        if not row["exact_match"]
    ][: max(0, show_failures)]

    if failures:
        print("\nSample failures:")
        for failure in failures:
            print(json.dumps(failure, ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()
    summary = asyncio.run(evaluate_system(args))
    backend_dir = Path(__file__).resolve().parents[1]
    output_path = args.output_path if args.output_path.is_absolute() else (backend_dir / args.output_path).resolve()
    write_output(summary, output_path)
    print_summary(summary, args.show_failures)
    print(f"\nRaw output written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
