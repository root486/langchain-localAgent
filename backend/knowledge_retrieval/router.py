"""知识库路由：判断一条用户消息是否应走知识库检索。

原实现（agent.py 的 KNOWLEDGE_SKILL_PATTERNS）只做正则匹配，既漏判也不太好调。
本模块用一次 LLM 二分类调用做判断，并保留两层兜底：
- STRONG_PATTERNS：显式提到知识库的快速通道（跳过 LLM，省一次调用 + 低延迟）
- FALLBACK_PATTERNS：LLM 调用失败/超时时的回退（= 改动前行为，保证不回归）

调用方式对齐 query_expander.py：LLM 通过 build_model 注入，便于测试与替换。
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Callable

# 显式提及知识库 → 直接进知识库，跳过 LLM 判断。
# 只保留"明确引用"的模式，不含 (查|检索).+文档 这类宽松匹配（正是误判来源）。
STRONG_PATTERNS = (
    re.compile(r"知识库"),
    re.compile(r"\bknowledge\b", re.IGNORECASE),
    re.compile(r"根据.+?(知识库|文档|资料)"),
    re.compile(r"\.(pdf|xlsx|xls|json)\b", re.IGNORECASE),
)

# LLM 调用失败时的回退正则 = 改动前的 KNOWLEDGE_SKILL_PATTERNS，逐条保留避免回归。
FALLBACK_PATTERNS = (
    re.compile(r"知识库"),
    re.compile(r"\bknowledge\b", re.IGNORECASE),
    re.compile(r"根据.+?(知识库|文档|资料)"),
    re.compile(r"(查|检索).+?(文档|资料|报告|白皮书)"),
    re.compile(r"\.(pdf|xlsx|xls|json)\b", re.IGNORECASE),
)

# 单次路由调用超时（秒）：超时按 LLM 失败处理，回退正则，不拖慢整条消息
ROUTER_TIMEOUT_SECONDS = 8

ROUTER_PROMPT = """你是检索意图判断器。判断用户问题是否应该从"本地知识库"中检索资料来回答。

本地知识库当前索引的文件清单（relative path，最多 40 条）：
{inventory}

判断规则：
- 问题涉及知识库内容（产品/流程/政策/FAQ/数据/报告等）→ use_knowledge=true
- 纯闲聊、代码问题、通用常识、当前时事，知识库显然没有资料 → use_knowledge=false
- 不确定时倾向 true（多检索一次成本低，答非所问更差）

只输出 JSON，不要任何其他文字：
{{"use_knowledge": true 或 false, "reason": "一句话中文理由"}}

问题: {question}
"""


def match_explicit(message: str) -> bool:
    """显式提及知识库 → 快速通道，直接进知识库。"""
    return any(pattern.search(message) for pattern in STRONG_PATTERNS)


def match_fallback(message: str) -> bool:
    """LLM 路由不可用时的回退判断，等价于改动前关键词匹配。"""
    return any(pattern.search(message) for pattern in FALLBACK_PATTERNS)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # 去掉 ```json / ``` 围栏
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_decision(text: str) -> bool | None:
    """从 LLM 输出里解析 use_knowledge。解析失败返回 None（由调用方回退正则）。"""
    text = _strip_code_fence(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # 兜底：模型可能夹带了前后缀，取首 { 到末 } 的子串再试
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    value = payload.get("use_knowledge")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


async def route(
    question: str,
    inventory: str,
    build_model: Callable,
    *,
    timeout: float = ROUTER_TIMEOUT_SECONDS,
) -> bool | None:
    """判断 question 是否应走知识库检索。

    - True / False：LLM 给出了明确结论
    - None：LLM 调用失败 / 超时 / 解析失败，由调用方回退关键词匹配
    """
    try:
        model = build_model()
        prompt = ROUTER_PROMPT.format(
            inventory=inventory or "(空)",
            question=question,
        )
        response = await asyncio.wait_for(model.ainvoke(prompt), timeout=timeout)
        text = str(getattr(response, "content", "")).strip()
        return _parse_decision(text)
    except Exception:
        return None
