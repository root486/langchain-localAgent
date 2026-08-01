"""查询扩展：RAG 检索前的 Query 改写优化。

解决的问题：用户口语化提问与知识库正式术语不匹配，导致检索召回不足。
通过 Router 判断问题类型，选择对应策略：
  A（模糊/口语化）→ 主改写 + 同义改写
  B（对比/多部分）→ 主改写 → 子查询分解
"""
from __future__ import annotations

import re
from typing import Callable

from langsmith import traceable

# ── Router ──
ROUTER_PROMPT = """判断用户问题类型，只返回 A 或 B：
A = 单一问题但表述模糊/口语化，只需补全语义
B = 包含多个部分/对比/前提条件，需拆开检索
只返回字母。

问题: {question}
类型:"""

# ── 主改写 ──
REFORMULATE_PROMPT = """你是检索查询优化器。将用户口语化问题改写为规范的检索语句。
规则：补全模糊指代和口语词，去掉"知识库""请帮我查"等触发前缀，只保留核心问题，保持简洁。
只返回改写后的单句。

用户问题: {question}
改写:"""

# ── 同义改写（A 用）──
SYNONYM_EXPAND_PROMPT = """你是检索查询优化器。为以下问题生成 2 个语义等价的变体。
换不同措辞或句式，使用核心实体的同义表达，每条是完整可检索的问句。
每行一条，不编号。

问题: {question}
变体:"""

# ── 子查询分解（B 用）──
DECOMPOSE_PROMPT = """你是检索查询优化器。将以下复杂问题拆为 2-3 个独立子问题。
每个子问题是完整可检索的单问句，合并后覆盖原问题所有维度。
每行一条，不编号。

示例:
问题: "退货和换货有什么区别"
子问题:
退货的流程和条件是什么
换货的流程和条件是什么

问题: {question}
子问题:"""


async def _call(prompt: str, build_model: Callable) -> str:
    try:
        model = build_model()
        response = await model.ainvoke(prompt)
        return str(response.content).strip()
    except Exception:
        return ""


def _parse_lines(text: str, limit: int = 3) -> list[str]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    lines = [re.sub(r"^\d+[\.\、\)]\s*", "", l).strip() for l in lines]
    return list(dict.fromkeys(lines))[:limit]


@traceable(
    run_type="chain",
    name="multi_query_expand",
    # build_model 是 callable，LangSmith 无法序列化，必须从 inputs 中剔除
    process_inputs=lambda i: {"question": i.get("question")},
)
async def expand(question: str, build_model: Callable) -> list[str]:
    """返回检索用 query 列表（原始 query 排第一位）。

    A: 主改写 → 同义改写(2条) → [原query, 主改写, 变体1, 变体2]
    B: 主改写 → 子查询(≤3条) → [原query, 子1, 子2, 子3]
    """
    route = await _call(ROUTER_PROMPT.format(question=question), build_model)
    route = route.strip().upper()[:1]
    if route not in ("A", "B"):
        route = "A"

    reformulated = await _call(
        REFORMULATE_PROMPT.format(question=question), build_model
    )
    base = reformulated or question

    if route == "A":
        expanded = await _call(SYNONYM_EXPAND_PROMPT.format(question=base), build_model)
        queries = [base] + _parse_lines(expanded, 2)
    else:
        decomposed = await _call(
            DECOMPOSE_PROMPT.format(question=base), build_model
        )
        queries = _parse_lines(decomposed, 3)

    return queries if queries else [question]
