#!/usr/bin/env python3
"""RAGAS 端到端评测脚本：对知识库检索 + 生成质量进行评测。

模型配置（自动从 .env 读取）：
  - Embedding 裁判：百炼 text-embedding-v4 → LangchainEmbeddingsWrapper → RAGAS
  - LLM 裁判：glm-5（智谱）→ LangchainLLMWrapper → RAGAS
  - 系统检索器：项目自身的 HybridRetriever（向量 + BM25 + RRF 融合）

运行：
    cd backend
    pip install ragas>=0.2.0
    python scripts/evaluate_ragas.py
    python scripts/evaluate_ragas.py --limit 20 --output storage/eval_outputs/ragas_result.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv

load_dotenv(BACKEND_DIR / ".env")

# ── RAGAS 0.4.x 兼容性补丁 ──────────────────────────────────
# RAGAS 内部导入 langchain_community.chat_models.vertexai，但新版 langchain 已移除此模块。
# 用 langchain-google-vertexai 桥接。
import langchain_community.chat_models
import langchain_google_vertexai
langchain_community.chat_models.vertexai = langchain_google_vertexai  # type: ignore[attr-defined]
sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai

from config import get_settings
from knowledge_retrieval.fusion import reciprocal_rank_fusion
from knowledge_retrieval.indexer import knowledge_indexer
from knowledge_retrieval.query_expander import expand as expand_queries
from knowledge_retrieval.reranker import rerank_evidences


# ---------------------------------------------------------------------------
# FAQ 数据加载
# ---------------------------------------------------------------------------

def load_faq_entries(faq_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(faq_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {faq_path}")

    entries: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            entries.append(
                {
                    "question": question,
                    "answer": answer,
                    "label": str(item.get("label", "")).strip(),
                    "url": str(item.get("url", "")).strip(),
                }
            )
    return entries


# ---------------------------------------------------------------------------
# 检索器初始化
# ---------------------------------------------------------------------------

def setup_retriever() -> KnowledgeIndexer:
    settings = get_settings()
    knowledge_indexer.configure(settings.backend_dir)

    status = knowledge_indexer.status()
    if not status.ready:
        knowledge_indexer.rebuild_index()
        status = knowledge_indexer.status()

    return knowledge_indexer


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------

def _build_query_expander_model():
    """为 Multi-Query 展开创建轻量 LLM（百炼 qwen-plus，与裁判同系列）。"""
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    return ChatOpenAI(
        model="qwen-plus",
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        temperature=0.0,
        max_tokens=256,
    )


async def retrieve_for_query(query: str, top_k: int = 4) -> list[str]:
    """走项目实际检索管线：Multi-Query 展开 → 向量 + BM25 → RRF 融合 → qwen3-rerank 精排。

    参数与 orchestrator.py 保持一致：
    - Multi-Query 展开（Router A/B，同生产）
    - 检索 top_k=4（同 hybrid_retriever）
    - RRF 融合 top_k=20（生产管线值）
    - Reranker 精排到最终数量
    """
    k = max(1, top_k)

    # ── Multi-Query 展开（与生产 orchestrator 一致）──
    all_queries = await expand_queries(query, _build_query_expander_model)
    all_queries.insert(0, query)  # 原问题也参与检索

    # ── 每条变体独立走向量 + BM25 ──
    all_vector: list = []
    all_bm25: list = []
    for q in all_queries:
        vector_evidences = knowledge_indexer.retrieve_vector(q, top_k=4)
        bm25_evidences = knowledge_indexer.retrieve_bm25(q, top_k=4)
        all_vector.extend(vector_evidences)
        all_bm25.extend(bm25_evidences)

    # RRF 融合取宽池（top_k=20，与生产管线一致）
    candidates = reciprocal_rank_fusion(
        [all_vector, all_bm25],
        top_k=20,
    )

    # RRF 结果不足时用原始检索结果补足
    seen: set[str] = set()
    for ev in candidates:
        seen.add(ev.snippet[:200])
    for ev in all_vector + all_bm25:
        key = ev.snippet[:200]
        if key not in seen:
            candidates.append(ev)
            seen.add(key)

    # qwen3-rerank 精排到最终需要的数量
    reranked = rerank_evidences(query, candidates, top_n=k)

    # 去重后返回
    deduped: list[str] = []
    dedup_keys: set[str] = set()
    for ev in reranked:
        key = ev.snippet[:200]
        if key not in dedup_keys:
            deduped.append(ev.snippet)
            dedup_keys.add(key)

    return deduped[:k]


# ---------------------------------------------------------------------------
# RAGAS 模型
# ---------------------------------------------------------------------------

def setup_ragas_models():
    """设置 RAGAS 评测用的 LLM（裁判用百炼 qwen-plus，与生成模型 GLM-5 不同）和 Embedding。"""
    from openai import OpenAI

    settings = get_settings()

    # 裁判 LLM：百炼 qwen-max（强指令遵循，结构化输出更稳）
    # 与生成模型 GLM-5 不同，避免自偏好偏差
    judge_model = "qwen-max"

    llm_client = OpenAI(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
    )

    emb_client = OpenAI(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
    )

    return llm_client, emb_client, judge_model


# ---------------------------------------------------------------------------
# 答案生成（用于评测 Faithfulness / AnswerCorrectness）
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
# RAGAS 评测
# ---------------------------------------------------------------------------

def run_ragas(
    dataset_records: list[dict[str, Any]],
    llm_client,
    emb_client,
    judge_model: str,
    with_generation: bool,
) -> Any:
    from ragas import evaluate
    from ragas import EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    # RAGAS 0.4.x: collections 指标不支持 evaluate()，用旧版 ragas.metrics
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    # LLM: LangchainLLMWrapper 支持 max_tokens（llm_factory 无法传入）
    from langchain_openai import ChatOpenAI
    # Embedding: 直接用 RagasOpenAIEmbeddings（有 embed_query）
    from ragas.embeddings.base import OpenAIEmbeddings as RagasOpenAIEmbeddings

    settings = get_settings()

    lc_judge = ChatOpenAI(
        model=judge_model,
        api_key=llm_client.api_key,
        base_url=str(llm_client.base_url),
        temperature=0.0,
        max_tokens=8192,
    )
    ragas_llm = LangchainLLMWrapper(lc_judge)
    # 不传 client，让 langchain 内部用 api_key/base_url 创建 client.embeddings
    # 否则 self.client 是原始 OpenAI 对象（没有 .create()），embed_documents 会报错
    ragas_embeddings = RagasOpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=llm_client.api_key,
        base_url=str(llm_client.base_url),
        check_embedding_ctx_length=False,  # 百炼非 OpenAI，绕过 tiktoken 长度检查
    )

    dataset = EvaluationDataset.from_list(dataset_records)

    # 先生成类指标放前面（问题指标优先测试，避免浪费时间）
    metrics = []
    if with_generation:
        metrics.extend([
            answer_relevancy,
            answer_correctness,
            faithfulness,
        ])
    # 检索类指标放后面
    metrics.extend([
        context_precision,
        context_recall,
    ])


    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def build_dataset(
    entries: list[dict[str, Any]],
    top_k: int,
    with_generation: bool,
    llm,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for entry in entries:
        query = entry["question"]
        reference = entry["answer"]
        contexts = await retrieve_for_query(query, top_k=top_k)

        record: dict[str, Any] = {
            "user_input": query,
            "retrieved_contexts": contexts,
            "reference": reference,
        }

        if with_generation:
            response = generate_answer(query, contexts, llm)
            record["response"] = response
        else:
            record["response"] = ""

        records.append(record)

    return records


def print_results(result: Any, with_generation: bool) -> dict[str, float]:
    """打印评测结果并返回扁平化的分数字典。"""
    METRIC_CN: dict[str, str] = {
        "context_precision": "上下文精确度",
        "context_recall": "上下文召回率",
        "faithfulness": "忠实度",
        "answer_relevancy": "答案相关性",
        "answer_correctness": "答案正确性",
    }

    scores: dict[str, float] = {}

    try:
        df = result.to_pandas()
        for col in df.columns:
            if col not in ("user_input", "retrieved_contexts", "response", "reference"):
                value = df[col].mean() if not df[col].empty else 0.0
                if isinstance(value, (int, float)):
                    scores[col] = float(value)
    except Exception:
        if isinstance(result, dict):
            for k, v in dict(result).items():
                if isinstance(v, (int, float)):
                    scores[k] = float(v)

    # ── Average results ──
    print("\n" + "=" * 55)
    print("  RAGAS Results (Average)")
    print("=" * 55)
    ORDER = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]
    for key in ORDER:
        if key in scores:
            cn = METRIC_CN.get(key, "")
            label = f"  {key} ({cn})"
            print(f"{label:<42s} {scores[key]:.4f}")
    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"  {'─' * 20}")
        print(f"  Overall Average{'':>22s} {avg:.4f}")
    print()

    # ── 逐条明细（CJK 宽度对齐）──
    try:
        df = result.to_pandas()
        score_cols = ["user_input"] + [c for c in ORDER if c in df.columns]
        _print_aligned_table(df, score_cols)
    except Exception:
        pass

    return scores


def _display_width(s: str) -> int:
    """CJK 字符算 2 个显示位，其余算 1。"""
    w = 0
    for ch in s:
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def _pad_right(s: str, width: int) -> str:
    """右对齐（数字用），补 CJK 宽度。"""
    dw = _display_width(s)
    return " " * max(0, width - dw) + s


def _pad_left(s: str, width: int, max_chars: int = 55) -> str:
    """左对齐（文本用），超长截断。"""
    if len(s) > max_chars:
        s = s[:max_chars - 1] + "…"
    dw = _display_width(s)
    return s + " " * max(0, width - dw)


def _print_aligned_table(df, cols: list[str]) -> None:
    # 列宽：问题文本 56 显示位，分数列 20 显示位
    Q_WIDTH = 56
    S_WIDTH = 20

    # 表头
    header_parts = []
    for c in cols:
        if c == "user_input":
            header_parts.append(_pad_left(c, Q_WIDTH))
        else:
            header_parts.append(_pad_right(c, S_WIDTH))
    print("  ".join(header_parts))

    # 分隔线
    sep_parts = []
    for c in cols:
        if c == "user_input":
            sep_parts.append("-" * Q_WIDTH)
        else:
            sep_parts.append("-" * S_WIDTH)
    print("  ".join(sep_parts))

    # 数据行
    for _, row in df[cols].iterrows():
        parts = []
        for c in cols:
            val = str(row[c])
            if c == "user_input":
                parts.append(_pad_left(val, Q_WIDTH))
            else:
                try:
                    parts.append(f"{float(val):>{S_WIDTH}.6f}")
                except (ValueError, TypeError):
                    parts.append(_pad_right(val, S_WIDTH))
        print("  ".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS 端到端评测")
    parser.add_argument(
        "--faq-path",
        type=Path,
        default=BACKEND_DIR / "knowledge" / "E-commerce Data" / "faq.json",
        help="FAQ JSON 文件路径",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="检索上下文数量（默认 6，与生产管线一致）",
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
        help="跳过答案生成，只评测检索质量",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="评测结果输出 JSON 路径",
    )
    args = parser.parse_args()

    # 1. 加载 FAQ ──────────────────────────────────────────
    faq_path = args.faq_path.resolve()
    if not faq_path.exists():
        print(f"[X] FAQ file not found: {faq_path}")
        return 1

    entries = load_faq_entries(faq_path)
    if args.limit > 0:
        entries = entries[: args.limit]

    setup_retriever()

    settings = get_settings()
    llm_client, emb_client, judge_model = setup_ragas_models()

    with_generation = not args.no_generation

    # 生成答案用 langchain LLM（项目已有）
    from langchain_openai import ChatOpenAI
    gen_llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.0,
        max_tokens=1024,
    )

    dataset_records = asyncio.run(
        build_dataset(
            entries,
            top_k=args.top_k,
            with_generation=with_generation,
            llm=gen_llm,
        )
    )

    result = run_ragas(
        dataset_records,
        llm_client,
        emb_client,
        judge_model,
        with_generation=with_generation,
    )

    # 6. 输出结果 ──────────────────────────────────────────
    scores = print_results(result, with_generation=with_generation)

    # 7. 保存 ──────────────────────────────────────────────
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # RAGAS 0.4.x: 转 pandas DataFrame 保存
        try:
            result_df = result.to_pandas()
            result_dict = result_df.to_dict(orient="records")
        except Exception:
            result_dict = {"raw": str(result)}

        output_payload = {
            "config": {
                "faq_path": str(faq_path),
                "top_k": args.top_k,
                "limit": args.limit or len(entries),
                "with_generation": with_generation,
                "generation_model": settings.llm_model,
                "judge_model": judge_model,
                "embedding_model": settings.embedding_model,
            },
            "scores": scores,
            "raw_result": result_dict,
            "dataset": [
                {
                    "user_input": r["user_input"],
                    "retrieved_contexts": r["retrieved_contexts"],
                    "response": r.get("response", ""),
                    "reference": r["reference"],
                }
                for r in dataset_records
            ],
        }
        output_path.write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[OK] 详细结果: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
