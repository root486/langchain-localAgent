from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_FAQ_PATH = Path("knowledge") / "E-commerce Data" / "faq.json"
DEFAULT_PREFIX = "\u8bf7\u5e2e\u6211\u5728\u77e5\u8bc6\u5e93\u67e5\u8be2\u5e76\u56de\u7b54\uff1a"
TARGET_FAQ_RELATIVE_PATH = "knowledge/E-commerce Data/faq.json"
DEFAULT_OUTPUT_PATH = Path("storage") / "eval_outputs" / "faq_agent_retrieval_results.json"
DEFAULT_RAGAS_DATASET_PATH = Path("storage") / "eval_outputs" / "faq_agent_ragas_dataset.jsonl"
DEFAULT_RAGAS_SUMMARY_PATH = Path("storage") / "eval_outputs" / "faq_agent_ragas_summary.json"
NO_CONTEXT_ID = "__no_retrieved_context__"
NO_CONTEXT_TEXT = "No supporting FAQ record was surfaced in the tool trace."


@dataclass(frozen=True)
class FaqEntry:
    record_id: str
    question: str
    answer: str
    label: str
    url: str


@dataclass
class EvalResult:
    index: int
    question: str
    prompt: str
    final_answer: str
    cleaned_answer: str
    reference_answer: str
    error: str
    tool_calls: list[dict[str, str]]
    routed_files: list[str]
    faq_file_routed: bool
    faq_entry_evidence_hit: bool
    answer_overlap: float
    retrieved_contexts: list[str]
    reference_contexts: list[str]
    retrieved_context_ids: list[str]
    reference_context_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate skill-rag routing by letting the model answer each FAQ question."
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
        help="Prefix added to every question to trigger knowledge-base retrieval skill.",
    )
    parser.add_argument(
        "--provider",
        help="Optional temporary LLM provider override for this evaluation process only.",
    )
    parser.add_argument(
        "--model",
        help="Optional temporary LLM model override for this evaluation process only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N entries. 0 means all entries.",
    )
    parser.add_argument(
        "--show-failures",
        type=int,
        default=10,
        help="How many failure samples to print.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the compact JSON report file, relative to backend/.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="How many questions to evaluate in parallel. Each question uses its own temporary session.",
    )
    parser.add_argument(
        "--ragas-mode",
        choices=("off", "basic", "full"),
        default="basic",
        help="Run no Ragas metrics, deterministic FAQ+Ragas metrics, or full LLM-based Ragas metrics.",
    )
    parser.add_argument(
        "--ragas-dataset-path",
        type=Path,
        default=DEFAULT_RAGAS_DATASET_PATH,
        help="Path to the exported Ragas dataset JSONL file, relative to backend/.",
    )
    parser.add_argument(
        "--ragas-summary-path",
        type=Path,
        default=DEFAULT_RAGAS_SUMMARY_PATH,
        help="Path to the Ragas summary JSON file, relative to backend/.",
    )
    return parser.parse_args()


def load_faq_entries(path: Path) -> list[FaqEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}, got {type(payload).__name__}")

    entries: list[FaqEntry] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        entries.append(
            FaqEntry(
                record_id=str(index),
                question=question,
                answer=answer,
                label=str(item.get("label", "")).strip(),
                url=str(item.get("url", "")).strip(),
            )
        )
    return entries


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff:/?.=&-]+", "", text)
    return text


def char_bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}


def overlap(left: str, right: str) -> float:
    normalized_left = normalize(left)
    normalized_right = normalize(right)
    if not normalized_left or not normalized_right:
        return 0.0
    left_bigrams = char_bigrams(normalized_left)
    right_bigrams = char_bigrams(normalized_right)
    if not left_bigrams or not right_bigrams:
        return 0.0
    return len(left_bigrams & right_bigrams) / len(left_bigrams | right_bigrams)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_process_paragraph(text: str) -> bool:
    normalized = collapse_whitespace(text)
    if not normalized:
        return False

    process_markers = (
        "我来帮",
        "我来为您",
        "让我",
        "首先让我",
        "现在我来",
        "根据知识库查询结果",
        "根据查询结果",
        "我在知识库中",
        "我找到",
        "我将从",
        "太好了",
        "首先",
    )
    return any(marker in normalized for marker in process_markers)


def clean_answer_for_eval(answer: str) -> str:
    text = answer.replace("\r\n", "\n").strip()
    if not text:
        return ""

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    trimmed_paragraphs: list[str] = []
    dropping_intro = True

    for paragraph in paragraphs:
        if dropping_intro and is_process_paragraph(paragraph):
            continue
        dropping_intro = False
        trimmed_paragraphs.append(paragraph)

    if not trimmed_paragraphs:
        trimmed_paragraphs = paragraphs

    cleaned_lines: list[str] = []
    for raw_line in "\n\n".join(trimmed_paragraphs).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if re.match(r"^#{1,6}\s*", line):
            continue
        if line.startswith("**来源文件"):
            continue
        if line.startswith("来源文件：") or line.startswith("来源："):
            continue
        if re.match(r"^[-*]\s*(文件路径|记录标签|相关链接|来源)\s*[：:]", line):
            continue
        if re.match(r"^(https?://|www\.)", line, re.IGNORECASE):
            continue

        line = re.sub(r"`([^`]*)`", r"\1", line)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"^\s*[-*]\s+", "", line)
        line = re.sub(r"^\s*\d+\.\s+", "", line)
        line = re.sub(r"^\s*[（(]来源[)）]\s*", "", line)
        line = line.strip()

        if len(line) <= 14 and not re.search(r"[。！？：:；;，,]", line):
            continue

        if line:
            cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or text


def sanitize_float(value: Any) -> Any:
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return sanitize_float(value)


def parse_tool_input(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {}


def join_tool_trace(tool_calls: list[dict[str, str]], final_answer: str) -> str:
    parts = [final_answer]
    for tool_call in tool_calls:
        parts.extend([tool_call.get("tool", ""), tool_call.get("input", ""), tool_call.get("output", "")])
    return "\n".join(parts)


def format_faq_context(entry: FaqEntry) -> str:
    parts = [
        f"FAQ ID: {entry.record_id}",
        f"Question: {entry.question}",
        f"Answer: {entry.answer}",
    ]
    if entry.label:
        parts.append(f"Label: {entry.label}")
    if entry.url:
        parts.append(f"URL: {entry.url}")
    return "\n".join(parts)


def extract_paths_from_text(text: str) -> list[str]:
    pattern = re.compile(
        r"(?:(?:skills|knowledge|workspace|memory|storage|api|graph|tools)/[^\s\"'`]+(?:\.[A-Za-z0-9_-]+)?)"
    )
    matches = pattern.findall(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for match in matches:
        normalized = match.rstrip(".,:;)]}>")
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def extract_routed_files(tool_calls: list[dict[str, str]], final_answer: str) -> list[str]:
    routed_files: list[str] = []
    seen: set[str] = set()

    def add_path(path: str) -> None:
        normalized = path.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        routed_files.append(normalized)

    for tool_call in tool_calls:
        payload = parse_tool_input(tool_call.get("input", ""))
        path_value = payload.get("path")
        if isinstance(path_value, str):
            add_path(path_value)

        for source in (tool_call.get("input", ""), tool_call.get("output", "")):
            for path in extract_paths_from_text(source):
                add_path(path)

    for path in extract_paths_from_text(final_answer):
        add_path(path)

    return routed_files


def extract_tool_output_snippets(
    tool_calls: list[dict[str, str]],
    *,
    limit: int = 3,
    max_chars: int = 600,
) -> list[str]:
    snippets: list[str] = []
    seen: set[str] = set()

    for tool_call in tool_calls:
        for field in ("output", "input"):
            raw = collapse_whitespace(str(tool_call.get(field, "")))
            if not raw:
                continue
            snippet = raw[:max_chars]
            if len(raw) > max_chars:
                snippet = f"{snippet}..."
            if snippet in seen:
                continue
            seen.add(snippet)
            snippets.append(snippet)
            if len(snippets) >= limit:
                return snippets

    return snippets


def detect_faq_route(tool_calls: list[dict[str, str]], final_answer: str) -> bool:
    trace = normalize(join_tool_trace(tool_calls, final_answer))
    return normalize(TARGET_FAQ_RELATIVE_PATH) in trace or "faqjson" in trace


def detect_entry_hit(entry: FaqEntry, tool_calls: list[dict[str, str]], final_answer: str) -> bool:
    trace = normalize(join_tool_trace(tool_calls, final_answer))
    question = normalize(entry.question)
    url = normalize(entry.url)
    answer = normalize(entry.answer[:200])

    if question and question in trace:
        return True
    if url and url in trace:
        return True
    if answer and answer in trace:
        return True
    return False


def score_entry_from_trace(trace: str, entry: FaqEntry) -> float:
    if not trace:
        return 0.0

    normalized_question = normalize(entry.question)
    normalized_label = normalize(entry.label)
    normalized_url = normalize(entry.url)
    normalized_answer = normalize(entry.answer[:200])

    score = 0.0
    if normalized_question and normalized_question in trace:
        score += 10.0
    if normalized_url and normalized_url in trace:
        score += 6.0
    if normalized_answer and normalized_answer in trace:
        score += 4.0
    if normalized_label and normalized_label in trace:
        score += 2.0

    return score


def extract_retrieved_faq_entries(
    entries: list[FaqEntry],
    tool_calls: list[dict[str, str]],
    final_answer: str,
    *,
    top_k: int = 3,
) -> list[FaqEntry]:
    trace = normalize(join_tool_trace(tool_calls, final_answer))
    ranked: list[tuple[FaqEntry, float]] = []

    for entry in entries:
        score = score_entry_from_trace(trace, entry)
        if score > 0:
            ranked.append((entry, score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return [entry for entry, _ in ranked[:top_k]]


def build_ragas_contexts(
    expected_entry: FaqEntry,
    entries: list[FaqEntry],
    tool_calls: list[dict[str, str]],
    final_answer: str,
    routed_files: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    matched_entries = extract_retrieved_faq_entries(entries, tool_calls, final_answer)
    reference_contexts = [format_faq_context(expected_entry)]
    reference_context_ids = [expected_entry.record_id]

    if matched_entries:
        return (
            [format_faq_context(entry) for entry in matched_entries],
            reference_contexts,
            [entry.record_id for entry in matched_entries],
            reference_context_ids,
        )

    snippets = extract_tool_output_snippets(tool_calls)
    if not snippets:
        routed = ", ".join(routed_files) or "none"
        snippets = [f"{NO_CONTEXT_TEXT} Routed files: {routed}."]

    return (
        snippets,
        reference_contexts,
        [NO_CONTEXT_ID],
        reference_context_ids,
    )


async def run_single_question(
    index: int,
    entry: FaqEntry,
    entries: list[FaqEntry],
    prefix: str,
    backend_dir: Path,
) -> EvalResult:
    from graph.agent import agent_manager

    session_manager = agent_manager.session_manager
    if session_manager is None:
        raise RuntimeError("Agent manager session manager is not initialized.")

    session = session_manager.create_session(title=f"eval-{index}")
    session_id = session["id"]
    prompt = f"{prefix}{entry.question}"
    tool_calls: list[dict[str, str]] = []
    final_answer = ""
    error = ""

    try:
        async for event in agent_manager.astream(prompt, history=[]):
            event_type = event.get("type", "")
            if event_type == "tool_start":
                tool_calls.append(
                    {
                        "tool": str(event.get("tool", "tool")),
                        "input": str(event.get("input", "")),
                        "output": "",
                    }
                )
            elif event_type == "tool_end":
                if tool_calls:
                    tool_calls[-1]["output"] = str(event.get("output", ""))
            elif event_type == "done":
                final_answer = str(event.get("content", "")).strip()
    except Exception as exc:  # pragma: no cover - runtime/API dependent
        error = str(exc)
    finally:
        session_manager.delete_session(session_id)

    cleaned_answer = clean_answer_for_eval(final_answer)
    faq_file_routed = detect_faq_route(tool_calls, final_answer)
    faq_entry_evidence_hit = detect_entry_hit(entry, tool_calls, final_answer)
    answer_overlap = overlap(cleaned_answer, entry.answer)
    routed_files = extract_routed_files(tool_calls, final_answer)
    (
        retrieved_contexts,
        reference_contexts,
        retrieved_context_ids,
        reference_context_ids,
    ) = build_ragas_contexts(entry, entries, tool_calls, final_answer, routed_files)

    return EvalResult(
        index=index,
        question=entry.question,
        prompt=prompt,
        final_answer=final_answer,
        cleaned_answer=cleaned_answer,
        reference_answer=entry.answer,
        error=error,
        tool_calls=tool_calls,
        routed_files=routed_files,
        faq_file_routed=faq_file_routed,
        faq_entry_evidence_hit=faq_entry_evidence_hit,
        answer_overlap=answer_overlap,
        retrieved_contexts=retrieved_contexts,
        reference_contexts=reference_contexts,
        retrieved_context_ids=retrieved_context_ids,
        reference_context_ids=reference_context_ids,
    )


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    backend_dir = Path(__file__).resolve().parents[1]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    faq_path = (backend_dir / args.faq_path).resolve()
    entries = load_faq_entries(faq_path)
    if args.limit > 0:
        entries = entries[: args.limit]

    from graph.agent import agent_manager

    agent_manager.initialize(backend_dir)

    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def guarded_run(index: int, entry: FaqEntry) -> EvalResult:
        async with semaphore:
            return await run_single_question(index, entry, entries, args.prefix, backend_dir)

    tasks = [
        asyncio.create_task(guarded_run(index, entry))
        for index, entry in enumerate(entries, start=1)
    ]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda item: item.index)

    total = len(results)
    success_results = [result for result in results if not result.error]
    error_results = [result for result in results if result.error]
    route_hits = [result for result in results if result.faq_file_routed]
    entry_hits = [result for result in results if result.faq_entry_evidence_hit]
    overlaps = [result.answer_overlap for result in results]

    return {
        "faq_path": str(faq_path),
        "output_path": str((backend_dir / args.output_path).resolve()),
        "ragas_dataset_path": str((backend_dir / args.ragas_dataset_path).resolve()),
        "ragas_summary_path": str((backend_dir / args.ragas_summary_path).resolve()),
        "prefix": args.prefix,
        "concurrency": max(1, args.concurrency),
        "ragas_mode": args.ragas_mode,
        "total": total,
        "success_count": len(success_results),
        "error_count": len(error_results),
        "file_route_accuracy": len(route_hits) / total if total else 0.0,
        "doc_correct_probability": len(entry_hits) / total if total else 0.0,
        "mean_answer_overlap": statistics.mean(overlaps) if overlaps else 0.0,
        "median_answer_overlap": statistics.median(overlaps) if overlaps else 0.0,
        "results": results,
    }


def build_ragas_dataset_payload(summary: dict[str, Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for result in summary["results"]:
        response = result.cleaned_answer.strip() or result.final_answer.strip()
        if not response and result.error:
            response = f"[ERROR] {result.error}"
        payload.append(
            {
                "index": result.index,
                "question": result.question,
                "user_input": result.prompt,
                "response": response,
                "raw_response": result.final_answer,
                "reference": result.reference_answer,
                "retrieved_contexts": result.retrieved_contexts,
                "reference_contexts": result.reference_contexts,
                "retrieved_context_ids": result.retrieved_context_ids,
                "reference_context_ids": result.reference_context_ids,
                "faq_file_routed": result.faq_file_routed,
                "faq_entry_evidence_hit": result.faq_entry_evidence_hit,
                "answer_overlap": result.answer_overlap,
                "error": result.error,
            }
        )
    return payload


def build_ragas_evaluator_llm(agent_manager: Any):
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(agent_manager._build_chat_model())


def build_ragas_embeddings():
    from config import get_settings
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    settings = get_settings()
    if not settings.embedding_api_key:
        return None

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
    )
    return LangchainEmbeddingsWrapper(embeddings)


def run_ragas(summary: dict[str, Any]) -> dict[str, Any] | None:
    ragas_mode = str(summary.get("ragas_mode", "off"))
    if ragas_mode == "off":
        return None

    try:
        from ragas import EvaluationDataset, evaluate as ragas_evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import (
            Faithfulness,
            IDBasedContextPrecision,
            IDBasedContextRecall,
            LLMContextPrecisionWithoutReference,
            NonLLMContextPrecisionWithReference,
            NonLLMContextRecall,
            ResponseRelevancy,
            SemanticSimilarity,
        )
    except ImportError as exc:
        return {
            "enabled": False,
            "error": f"Ragas dependencies are not installed: {exc}",
        }

    dataset_payload = build_ragas_dataset_payload(summary)
    samples = [
        SingleTurnSample(
            user_input=item["user_input"],
            response=item["response"],
            reference=item["reference"],
            retrieved_contexts=item["retrieved_contexts"],
            reference_contexts=item["reference_contexts"],
            retrieved_context_ids=item["retrieved_context_ids"],
            reference_context_ids=item["reference_context_ids"],
        )
        for item in dataset_payload
    ]

    metrics: list[Any] = [
        IDBasedContextPrecision(),
        IDBasedContextRecall(),
        NonLLMContextPrecisionWithReference(),
        NonLLMContextRecall(),
    ]
    skipped_metrics: list[str] = []
    ragas_llm = None
    ragas_embeddings = None

    if ragas_mode == "full":
        try:
            from graph.agent import agent_manager

            ragas_llm = build_ragas_evaluator_llm(agent_manager)
        except Exception as exc:
            reason = f"missing evaluator llm: {exc}"
            skipped_metrics.extend(
                [
                    f"faithfulness ({reason})",
                    f"llm_context_precision_without_reference ({reason})",
                    f"answer_relevancy ({reason})",
                ]
            )

        try:
            ragas_embeddings = build_ragas_embeddings()
        except Exception as exc:
            ragas_embeddings = None
            reason = f"missing evaluator embeddings: {exc}"
            skipped_metrics.extend(
                [
                    f"semantic_similarity ({reason})",
                    f"answer_relevancy ({reason})",
                ]
            )

        if ragas_llm is not None:
            metrics.extend([Faithfulness(), LLMContextPrecisionWithoutReference()])
        if ragas_embeddings is not None:
            metrics.append(SemanticSimilarity())
        if ragas_llm is not None and ragas_embeddings is not None:
            metrics.append(ResponseRelevancy())

    dataset = EvaluationDataset(samples=samples)
    evaluation = ragas_evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
        show_progress=True,
    )

    summary_scores = {
        key: sanitize_float(value)
        for key, value in getattr(evaluation, "_repr_dict", {}).items()
    }
    per_sample_scores = [
        {key: json_ready(value) for key, value in row.items()}
        for row in evaluation.to_pandas().to_dict(orient="records")
    ]

    return {
        "enabled": True,
        "mode": ragas_mode,
        "dataset_payload": dataset_payload,
        "metrics": [metric.name for metric in metrics],
        "skipped_metrics": skipped_metrics,
        "summary_scores": summary_scores,
        "per_sample_scores": per_sample_scores,
    }


def write_report(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "question": result.question,
            "final_answer": result.final_answer,
        }
        for result in summary["results"]
    ]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_ragas_outputs(summary: dict[str, Any], ragas_result: dict[str, Any] | None) -> None:
    if not ragas_result:
        return

    dataset_path = Path(summary["ragas_dataset_path"])
    summary_path = Path(summary["ragas_summary_path"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in ragas_result.get("dataset_payload", []):
            handle.write(json.dumps(json_ready(row), ensure_ascii=False))
            handle.write("\n")

    summary_payload = {
        "enabled": bool(ragas_result.get("enabled", True)),
        "mode": ragas_result.get("mode"),
        "metrics": ragas_result.get("metrics", []),
        "skipped_metrics": ragas_result.get("skipped_metrics", []),
        "summary_scores": ragas_result.get("summary_scores", {}),
        "error": ragas_result.get("error", ""),
    }
    summary_path.write_text(
        json.dumps(json_ready(summary_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_summary(summary: dict[str, Any], show_failures: int) -> None:
    total = int(summary["total"])
    success_count = int(summary["success_count"])
    error_count = int(summary["error_count"])
    file_route_accuracy = float(summary["file_route_accuracy"])
    doc_correct_probability = float(summary["doc_correct_probability"])
    mean_answer_overlap = float(summary["mean_answer_overlap"])
    median_answer_overlap = float(summary["median_answer_overlap"])

    print(f"FAQ file: {summary['faq_path']}")
    print(f"Report JSON: {summary['output_path']}")
    print(f"Ragas mode: {summary['ragas_mode']}")
    if summary["ragas_mode"] != "off":
        print(f"Ragas dataset JSONL: {summary['ragas_dataset_path']}")
        print(f"Ragas summary JSON: {summary['ragas_summary_path']}")
    print(f"Concurrency: {summary['concurrency']}")
    print(f"Total samples: {total}")
    print(f"Successful runs: {success_count}")
    print(f"Errored runs: {error_count}")
    print(
        "Retrieval precision (model routed to faq.json): "
        f"{file_route_accuracy:.2%} ({round(file_route_accuracy * total)}/{total})"
    )
    print(
        "Probability recalled document is correct: "
        f"{doc_correct_probability:.2%} ({round(doc_correct_probability * total)}/{total})"
    )
    print(f"Mean answer overlap with gold answer: {mean_answer_overlap:.2%}")
    print(f"Median answer overlap with gold answer: {median_answer_overlap:.2%}")

    failures: list[dict[str, Any]] = []
    for result in summary["results"]:
        if result.error or not result.faq_file_routed or not result.faq_entry_evidence_hit:
            failures.append(
                {
                    "index": result.index,
                    "prompt": result.prompt,
                    "error": result.error,
                    "faq_file_routed": result.faq_file_routed,
                    "faq_entry_evidence_hit": result.faq_entry_evidence_hit,
                    "answer_overlap": round(result.answer_overlap, 4),
                    "tool_count": len(result.tool_calls),
                    "routed_files": result.routed_files,
                    "final_answer_preview": result.final_answer[:300],
                }
            )

    if failures[: max(0, show_failures)]:
        print("\nSample failures:")
        for item in failures[: max(0, show_failures)]:
            print(json.dumps(item, ensure_ascii=False, indent=2))
    else:
        print("\nNo routing failures found in this run.")


def print_ragas_summary(ragas_result: dict[str, Any] | None) -> None:
    if not ragas_result:
        return

    if not ragas_result.get("enabled", True):
        print("\nRagas evaluation was skipped.")
        if ragas_result.get("error"):
            print(ragas_result["error"])
        return

    print("\nRagas summary:")
    for metric_name in ragas_result.get("metrics", []):
        value = ragas_result.get("summary_scores", {}).get(metric_name)
        if value is None:
            print(f"- {metric_name}: null")
        elif isinstance(value, (int, float)):
            print(f"- {metric_name}: {value:.4f}")
        else:
            print(f"- {metric_name}: {value}")

    skipped_metrics = ragas_result.get("skipped_metrics", [])
    if skipped_metrics:
        print("Skipped metrics:")
        for item in skipped_metrics:
            print(f"- {item}")


def main() -> int:
    args = parse_args()
    summary = asyncio.run(evaluate(args))
    ragas_result = run_ragas(summary)
    write_report(summary, Path(summary["output_path"]))
    write_ragas_outputs(summary, ragas_result)
    print_summary(summary, show_failures=args.show_failures)
    print_ragas_summary(ragas_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
