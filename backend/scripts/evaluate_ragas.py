#!/usr/bin/env python3
"""RAGAS 端到端评测（两层）：确定性检索评估 + RAGAS 答案质量评估。

第一层（检索，零 LLM）：
  对每条 FAQ 走生产检索管线（Multi-Query → vector + BM25 → RRF → qwen3-rerank），
  用 `record_id` ↔ locator("记录 N") 判定命中，算 recall@k / precision@k / MRR / nDCG@k，
  并按阶段拆（vector / bm25 / fused / final）定位失败环节。

第二层（答案质量，RAGAS 0.4.3 LLM 判定）：
  受控生成答案后评测 Faithfulness / AnswerRelevancy / AnswerCorrectness。
  RAGAS 的 context_recall / context_precision 已被第一层二进制指标替代
  （FAQ 场景下标准答案就在知识库里，那两个指标恒近 1.0，是自证循环）。

运行（从 backend/ 目录）：
    python scripts/evaluate_ragas.py                         # 全量两层（费 token）
    python scripts/evaluate_ragas.py --limit 2               # 省 token 抽查
    python scripts/evaluate_ragas.py --no-generation         # 只跑检索层（零 LLM）
    python scripts/evaluate_ragas.py --no-generation --no-multi-query   # 完全离线
    python scripts/evaluate_ragas.py --output storage/eval_outputs/ragas_result.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 中文输出

from dotenv import load_dotenv

load_dotenv(BACKEND_DIR / ".env")

# ── RAGAS 0.4.3 ↔ langchain-community 1.x 兼容补丁 ──────────────
# 必须先于任何 `import ragas` 导入（见 _ragas_compat.py 注释）。
import _ragas_compat  # noqa: F401

from config import get_settings
from knowledge_retrieval.fusion import reciprocal_rank_fusion
from knowledge_retrieval.indexer import knowledge_indexer
from knowledge_retrieval.query_expander import expand as expand_queries
from knowledge_retrieval.reranker import rerank_evidences


# ---------------------------------------------------------------------------
# FAQ 数据加载
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FaqEntry:
    record_id: str  # 与 indexer._split_json 的 record_id 规则一致
    question: str
    answer: str
    label: str
    url: str


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


# ---------------------------------------------------------------------------
# 检索（确定性，零 LLM）
# ---------------------------------------------------------------------------

def _build_query_expander_model():
    """Multi-Query 展开用轻量 LLM（百炼 qwen-plus，与裁判同 provider）。"""
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    return ChatOpenAI(
        model="qwen-plus",
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        temperature=0.0,
        max_tokens=256,
    )


def _dedupe(evidences: list) -> list:
    """按 (source_path, locator) 去重（同一条 FAQ chunk 可能被多个查询变体命中）。"""
    seen: set[tuple[str, str]] = set()
    out: list = []
    for ev in evidences:
        key = (ev.source_path, ev.locator)
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


def _dedupe_fill(base: list, pool: list) -> list:
    """RRF 融合结果不足时用原始检索结果补足（与生产 orchestrator 一致）。"""
    seen: set[tuple[str, str]] = {(ev.source_path, ev.locator) for ev in base}
    out = list(base)
    for ev in pool:
        key = (ev.source_path, ev.locator)
        if key not in seen:
            out.append(ev)
            seen.add(key)
    return out


async def retrieve_pipeline(
    query: str, *, top_k: int, use_multi_query: bool
) -> dict[str, list]:
    """返回 {vector, bm25, fused, final} 各阶段 Evidence 列表（保留 locator）。

    参数与生产 orchestrator.py 保持一致：
    - Multi-Query 展开（Router A/B，同生产）
    - 每条变体 vector + bm25 top_k=4
    - RRF 融合宽池 top_k=20
    - Rerank 精排到最终数量 top_k
    """
    all_queries = [query]
    if use_multi_query:
        variants = await expand_queries(query, _build_query_expander_model)
        all_queries.extend(variants)  # 原问题已在首位（与 orchestrator 一致）

    all_vector: list = []
    all_bm25: list = []
    for q in all_queries:
        all_vector.extend(knowledge_indexer.retrieve_vector(q, top_k=4))
        all_bm25.extend(knowledge_indexer.retrieve_bm25(q, top_k=4))

    # RRF 融合取宽池（top_k=20，与生产管线一致）
    fused = reciprocal_rank_fusion([all_vector, all_bm25], top_k=20)
    fused = _dedupe_fill(fused, all_vector + all_bm25)

    # qwen3-rerank 精排到最终数量（失败自动降级为 fused[:top_n]）
    final = rerank_evidences(query, fused, top_n=top_k)

    return {
        "vector": _dedupe(all_vector),
        "bm25": _dedupe(all_bm25),
        "fused": fused,
        "final": _dedupe(final)[:top_k],
    }


# ---------------------------------------------------------------------------
# 命中判定与检索指标
# ---------------------------------------------------------------------------

def _hit(evidence: Any, record_id: str) -> bool:
    """判断检索结果是否命中指定 FAQ 记录（locator 形如 "记录 N" 或 "记录 N (片段 x/y)"）。"""
    return f"记录 {record_id}" in (evidence.locator or "")


def _stage_metrics(evidences: list, record_id: str, k: int) -> dict[str, float]:
    """单条 FAQ 在某阶段的指标。正确 chunk 唯一，故 recall 是 0/1。"""
    rank = next(
        (i + 1 for i, ev in enumerate(evidences) if _hit(ev, record_id)), None
    )
    hit = rank is not None and rank <= k
    return {
        "recall_at_k": 1.0 if hit else 0.0,
        "mrr": 1.0 / rank if rank else 0.0,
        "ndcg_at_k": 1.0 / math.log2(rank + 1) if hit else 0.0,
        "precision_at_k": 1.0 / k if hit else 0.0,
    }


def aggregate_retrieval(meta_records: list[dict], top_k: int) -> dict[str, Any]:
    """汇总检索层：各阶段均值 + top1/topk 命中数 + 跨阶段归因 + 失败样例。"""
    stages = ("vector", "bm25", "fused", "final")
    agg: dict[str, Any] = {"total": len(meta_records), "stages": {}}

    for stage in stages:
        metrics = [r["stage_metrics"][stage] for r in meta_records]
        agg["stages"][stage] = {
            "recall_at_k": _mean(metrics, "recall_at_k"),
            "mrr": _mean(metrics, "mrr"),
            "ndcg_at_k": _mean(metrics, "ndcg_at_k"),
            "precision_at_k": _mean(metrics, "precision_at_k"),
            "top1_hits": sum(1 for m in metrics if m["mrr"] == 1.0),
            "topk_hits": sum(1 for m in metrics if m["recall_at_k"] == 1.0),
        }

    # 跨阶段归因：vector / bm25 命中了但 final 丢（RRF 或 rerank 环节丢的）
    vector_hit_final_miss = sum(
        1 for r in meta_records if r["vector_hit"] and not r["final_hit"]
    )
    bm25_hit_final_miss = sum(
        1 for r in meta_records if r["bm25_hit"] and not r["final_hit"]
    )
    agg["lost_between_stages"] = {
        "vector_hit_final_miss": vector_hit_final_miss,
        "bm25_hit_final_miss": bm25_hit_final_miss,
    }

    # 失败样例（final 未命中）
    failures = []
    for r in meta_records:
        if not r["final_hit"]:
            failures.append(
                {
                    "record_id": r["record_id"],
                    "question": r["question"],
                    "label": r["label"],
                    "vector_hit": r["vector_hit"],
                    "bm25_hit": r["bm25_hit"],
                    "fused_hit": r["fused_hit"],
                    "top1_locator": r["final_locators"][0]
                    if r["final_locators"]
                    else "",
                    "top1_source": r["final_sources"][0]
                    if r["final_sources"]
                    else "",
                }
            )
    agg["failures"] = failures

    return agg


def _mean(metrics: list[dict], key: str) -> float:
    if not metrics:
        return 0.0
    return sum(m.get(key, 0.0) for m in metrics) / len(metrics)


# ---------------------------------------------------------------------------
# 答案生成（受控，供 RAGAS 评测）
# ---------------------------------------------------------------------------

_SYSTEM_GEN_PROMPT = (
    "你是一个客服助手。请严格根据以下参考资料回答用户问题。"
    "只使用参考资料中提供的信息，不要编造或猜测。"
    "如果参考资料中没有相关信息，请直接说「根据现有资料无法回答」。"
)


def generate_answer(query: str, contexts: list[str], llm) -> str:
    if not contexts:
        return "根据现有资料无法回答。"

    context_text = "\n\n---\n\n".join(
        f"[来源 {i}]\n{ctx}" for i, ctx in enumerate(contexts, 1)
    )

    prompt = (
        f"{_SYSTEM_GEN_PROMPT}\n\n"
        f"【参考资料】\n{context_text}\n\n"
        f"【用户问题】\n{query}\n\n"
        "【回答】"
    )

    from langchain_core.messages import HumanMessage

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        return f"[生成失败: {exc}]"


# ---------------------------------------------------------------------------
# 数据集构建（RAGAS 记录 + 检索元数据分开，避免额外字段触发校验）
# ---------------------------------------------------------------------------

async def build_dataset(
    entries: list[FaqEntry],
    *,
    top_k: int,
    use_multi_query: bool,
    with_generation: bool,
    gen_llm,
) -> tuple[list[dict], list[dict]]:
    ragas_records: list[dict] = []
    meta_records: list[dict] = []

    for entry in entries:
        stages = await retrieve_pipeline(
            entry.question, top_k=top_k, use_multi_query=use_multi_query
        )
        final_snippets = [ev.snippet for ev in stages["final"]]

        stage_metrics = {
            stage: _stage_metrics(evs, entry.record_id, top_k)
            for stage, evs in stages.items()
        }

        meta_records.append(
            {
                "record_id": entry.record_id,
                "question": entry.question,
                "label": entry.label,
                "vector_hit": stage_metrics["vector"]["recall_at_k"] == 1.0,
                "bm25_hit": stage_metrics["bm25"]["recall_at_k"] == 1.0,
                "fused_hit": stage_metrics["fused"]["recall_at_k"] == 1.0,
                "final_hit": stage_metrics["final"]["recall_at_k"] == 1.0,
                "stage_metrics": stage_metrics,
                "final_locators": [ev.locator for ev in stages["final"]],
                "final_sources": [ev.source_path for ev in stages["final"]],
            }
        )

        rec: dict[str, Any] = {
            "user_input": entry.question,
            "retrieved_contexts": final_snippets,
            "reference": entry.answer,
        }
        if with_generation:
            rec["response"] = generate_answer(entry.question, final_snippets, gen_llm)
        else:
            rec["response"] = ""
        ragas_records.append(rec)

    return ragas_records, meta_records


# ---------------------------------------------------------------------------
# RAGAS 答案质量层
# ---------------------------------------------------------------------------

def run_ragas_answer_metrics(
    ragas_records: list[dict], judge_model: str
) -> Any:
    """用 RAGAS 0.4.3 评测答案质量三指标。

    注意（实测）：0.4.3 的 `evaluate()` 只接受 `ragas.metrics` 懒加载的
    旧体系指标实例（类继承 MetricWithLLM→Metric）；`ragas.metrics.collections`
    的新体系类（继承 BaseMetric）会被 `evaluate()` 拒绝。因此这里走
    `ragas.metrics` 的预实例化指标 + `evaluate(llm=, embeddings=)` 绑定，
    与旧脚本已验证路径一致（DeprecationWarning 仅噪音，v1.0 才移除）。
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings.base import OpenAIEmbeddings as RagasOpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    settings = get_settings()

    # 屏蔽两个 DeprecationWarning：指标导入路径 + LangchainLLMWrapper
    # （两者都是 ragas 0.4.3 过渡期的噪音，替换路径实测不可靠：
    #   collections 类体系被 evaluate() 拒绝；llm_factory(InstructorLLM)
    #   对 qwen-max 结构化输出解析失败——见 git 历史，勿再试）
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Importing .* from 'ragas.metrics'"
        )
        warnings.filterwarnings(
            "ignore", message="LangchainLLMWrapper is deprecated.*"
        )
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            faithfulness,
        )
        lc_judge = ChatOpenAI(
            model=judge_model,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            temperature=0.0,
            max_tokens=8192,
        )
        ragas_llm = LangchainLLMWrapper(lc_judge)

    ragas_embeddings = RagasOpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        check_embedding_ctx_length=False,  # 百炼非 OpenAI，绕过 tiktoken 长度检查
    )

    dataset = EvaluationDataset.from_list(ragas_records)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, answer_correctness],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return result


def extract_answer_scores(result: Any) -> tuple[dict[str, float], list[dict]]:
    """把 RAGAS 结果转成 {指标: 均值} 和逐条明细。"""
    df = result.to_pandas()
    metric_cols = [
        c
        for c in df.columns
        if c not in ("user_input", "retrieved_contexts", "response", "reference")
    ]

    scores: dict[str, float] = {}
    for c in metric_cols:
        if df[c].empty:
            continue
        try:
            scores[c] = float(df[c].mean())
        except (TypeError, ValueError):
            continue

    if metric_cols:
        detail = df[["user_input"] + metric_cols].to_dict(orient="records")
    else:
        detail = []
    return scores, detail


# ---------------------------------------------------------------------------
# 打印（CJK 对齐）
# ---------------------------------------------------------------------------

def _display_width(s: str) -> int:
    w = 0
    for ch in s:
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def _pad_right(s: str, width: int) -> str:
    dw = _display_width(s)
    return " " * max(0, width - dw) + s


def _pad_left(s: str, width: int, max_chars: int = 55) -> str:
    if len(s) > max_chars:
        s = s[: max_chars - 1] + "…"
    dw = _display_width(s)
    return s + " " * max(0, width - dw)


def print_retrieval(agg: dict, top_k: int, show_failures: int) -> None:
    print("\n" + "=" * 60)
    print(f"  检索层（确定性二进制指标，top-{top_k}）")
    print("=" * 60)
    print(f"  样本数: {agg['total']}")
    header = (
        f"  {'阶段':<8}{'recall@k':>10}{'MRR':>8}{'nDCG@k':>9}"
        f"{'prec@k':>9}{'top1':>8}{'topk':>7}"
    )
    print(header)
    print("  " + "-" * 55)
    for stage in ("vector", "bm25", "fused", "final"):
        s = agg["stages"][stage]
        print(
            f"  {stage:<8}{s['recall_at_k']:>10.3f}{s['mrr']:>8.3f}"
            f"{s['ndcg_at_k']:>9.3f}{s['precision_at_k']:>9.3f}"
            f"{s['top1_hits']:>8}{s['topk_hits']:>7}"
        )
    lost = agg.get("lost_between_stages", {})
    print(
        f"  跨阶段丢失: vector命中但final丢={lost.get('vector_hit_final_miss', 0)}"
        f"  bm25命中但final丢={lost.get('bm25_hit_final_miss', 0)}"
    )

    failures = agg.get("failures", [])[: max(0, show_failures)]
    if failures:
        print(f"\n  检索失败样例（final 未命中，前 {len(failures)} 条）:")
        for f in failures:
            print(f"    Q: {f['question']}")
            print(
                f"      期望记录 {f['record_id']} | vector命中={f['vector_hit']}"
                f" | bm25命中={f['bm25_hit']} | fused命中={f['fused_hit']}"
            )
            print(f"      top1: {f['top1_locator']} ({f['top1_source']})")
    else:
        print("\n  final 阶段全部命中，无失败样例。")


def print_answer(answer_scores: dict[str, float], detail: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("  答案质量层（RAGAS，LLM 判定）")
    print("=" * 60)
    for k in ("faithfulness", "answer_relevancy", "answer_correctness"):
        print(f"  {k:<22} {answer_scores.get(k, 0.0):.4f}")

    if detail:
        # 逐条明细（CJK 宽度对齐）
        Q_WIDTH = 56
        S_WIDTH = 16
        metric_cols = [c for c in detail[0] if c != "user_input"]
        print()
        header_parts = [_pad_left("user_input", Q_WIDTH)]
        header_parts += [_pad_right(c, S_WIDTH) for c in metric_cols]
        print("  ".join(header_parts))
        print("  ".join(["-" * Q_WIDTH] + ["-" * S_WIDTH] * len(metric_cols)))
        for row in detail:
            parts = [_pad_left(str(row["user_input"]), Q_WIDTH)]
            for c in metric_cols:
                try:
                    parts.append(f"{float(row[c]):>{S_WIDTH}.4f}")
                except (ValueError, TypeError):
                    parts.append(_pad_right(str(row[c]), S_WIDTH))
            print("  ".join(parts))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAGAS 两层评测：确定性检索（recall@k/MRR/nDCG）+ 答案质量（LLM 判定）"
    )
    parser.add_argument(
        "--faq-path",
        type=Path,
        default=Path("knowledge") / "E-commerce Data" / "faq.json",
        help="FAQ JSON 路径（相对 backend/）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="最终检索上下文数量（默认 4，与生产管线一致）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="限制评测条数（0 = 全部）",
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="只跑检索层（零 LLM，跳过 RAGAS 答案质量）",
    )
    parser.add_argument(
        "--no-multi-query",
        action="store_true",
        help="跳过 LLM 查询改写，只用原始问题检索（离线快速验证）",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen-max",
        help="RAGAS 裁判 LLM（默认 qwen-max，百炼）",
    )
    parser.add_argument(
        "--gen-model",
        default=None,
        help="答案生成 LLM（默认取 .env 的 llm_model）",
    )
    parser.add_argument(
        "--show-failures",
        type=int,
        default=10,
        help="最多打印多少条检索失败样例",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="评测结果输出 JSON 路径",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    # 1. 索引就绪 ──────────────────────────────────────────────
    knowledge_indexer.configure(settings.backend_dir)
    status = knowledge_indexer.status()
    if not status.ready:
        print("[评测] 索引未就绪，正在重建...")
        knowledge_indexer.rebuild_index()
        status = knowledge_indexer.status()
    if not status.vector_ready and not status.bm25_ready:
        print("[评测] 警告：向量/BM25 索引均不可用，检索层结果为空")

    # 2. 加载 FAQ ──────────────────────────────────────────────
    faq_path = (settings.backend_dir / args.faq_path).resolve()
    if not faq_path.exists():
        print(f"[X] FAQ file not found: {faq_path}")
        return 1
    entries = load_faq_entries(faq_path)
    if args.limit > 0:
        entries = entries[: args.limit]

    top_k = max(1, args.top_k)
    use_multi_query = not args.no_multi_query
    with_generation = not args.no_generation

    # 3. 生成 LLM（仅答案质量层需要）────────────────────────────
    gen_llm = None
    if with_generation:
        from langchain_openai import ChatOpenAI

        gen_model = args.gen_model or settings.llm_model
        gen_llm = ChatOpenAI(
            model=gen_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=0.0,
            max_tokens=1024,
        )

    # 4. 构建数据集（检索 + 生成）───────────────────────────────
    ragas_records, meta_records = asyncio.run(
        build_dataset(
            entries,
            top_k=top_k,
            use_multi_query=use_multi_query,
            with_generation=with_generation,
            gen_llm=gen_llm,
        )
    )

    # 5. 检索层（确定性，零 LLM）───────────────────────────────
    retrieval_layer = aggregate_retrieval(meta_records, top_k)
    print_retrieval(retrieval_layer, top_k, args.show_failures)

    # 6. 答案质量层（RAGAS）─────────────────────────────────────
    answer_scores: dict[str, float] = {}
    answer_detail: list[dict] = []
    if with_generation:
        result = run_ragas_answer_metrics(ragas_records, args.judge_model)
        answer_scores, answer_detail = extract_answer_scores(result)
        print_answer(answer_scores, answer_detail)

    # 7. 输出 ──────────────────────────────────────────────────
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "faq_path": str(faq_path),
                "top_k": top_k,
                "limit": args.limit or len(entries),
                "with_generation": with_generation,
                "multi_query": use_multi_query,
                "generation_model": args.gen_model or settings.llm_model,
                "judge_model": args.judge_model,
                "embedding_model": settings.embedding_model,
            },
            "retrieval_layer": retrieval_layer,
            "answer_layer": answer_scores,
            "answer_detail": answer_detail,
            "dataset": [
                {
                    "user_input": r["user_input"],
                    "retrieved_contexts": r["retrieved_contexts"],
                    "response": r.get("response", ""),
                    "reference": r["reference"],
                }
                for r in ragas_records
            ],
            "retrieval_meta": meta_records,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[OK] 详细结果: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
