from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import tiktoken
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langsmith import tracing_context
from pydantic import BaseModel, Field

from graph.agent import agent_manager
from graph.memory_store import memory_store
from graph.prompt_builder import build_system_prompt
from config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# 达到模型上下文窗口的 80% 时自动触发压缩
AUTO_COMPRESS_RATIO = 0.8
_token_encoder = tiktoken.get_encoding("cl100k_base")
# 长期记忆自动抽取：距上次抽取的新消息 ≥ 该条数才后台抽取一次（控制 LLM 调用频率）
MEMORY_EXTRACT_MIN_MESSAGES = 6


def _count_tokens(text: str) -> int:
    return len(_token_encoder.encode(text or ""))


def _total_tokens(messages: list[dict]) -> int:
    """计算发给 LLM 的全部内容 token 数：system prompt + 消息全量"""
    base_dir = agent_manager.base_dir
    system_text = (
        build_system_prompt(base_dir)
        if base_dir else ""
    )
    # 整段序列化，包含 content + tool_calls + retrieval_steps
    msg_text = json.dumps(messages, ensure_ascii=False, default=str)
    return _count_tokens(system_text) + _count_tokens(msg_text)


def _history_tokens(messages: list[dict]) -> int:
    """只统计会话历史（含 tool_calls / retrieval_steps）的 token 数，不含 system prompt。"""
    return _count_tokens(json.dumps(messages, ensure_ascii=False, default=str))


# 后台压缩任务引用集：持有引用防 GC（否则任务挂起 await 时可能被销毁），完成后自动移除
_bg_tasks: set[asyncio.Task] = set()


def _schedule_compression(session_id: str) -> None:
    """把自动压缩调度到后台任务，避免阻塞 SSE done 事件（尾部不再等摘要 LLM 调用）。"""
    task = asyncio.create_task(_run_compression(session_id))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


async def _run_compression(session_id: str) -> None:
    """后台压缩：固定历史 token 预算触发 + 摘要链二次折叠。

    调度与执行分离：执行时重新读 record 并重判触发条件，避免用调度那一刻的过期状态。
    """
    try:
        session_manager = agent_manager.session_manager
        if session_manager is None:
            return
        settings = get_settings()
        record = session_manager.get_history(session_id)
        messages = record.get("messages", [])
        existing = (record.get("compressed_context") or "").strip()

        # 预算触发（固定历史 token，摘要链也计入） + 80% 硬上限兜底（system prompt 可能很大，唯一防线）
        history_tokens = _history_tokens(messages) + _count_tokens(existing)
        budget_hit = (
            history_tokens > settings.auto_compress_token_limit
            and len(messages) >= 4
        )
        hard_hit = (
            _total_tokens(messages) > int(settings.max_context_tokens * AUTO_COMPRESS_RATIO)
            and len(messages) >= 2
        )
        if not (budget_hit or hard_hit):
            return

        # 归档前一半，保留最后两条（当前回合永不归档）
        n = min(len(messages) // 2, max(4, len(messages) - 2))
        summary = await agent_manager.summarize_history(messages[:n])

        # 摘要链追加后判定：超限则一次调用折叠为单段，避免链无界增长
        context = f"{existing}\n---\n{summary}".strip() if existing else summary.strip()
        if _count_tokens(context) > settings.summary_chain_token_limit:
            context = await agent_manager.recompress_context(context)

        session_manager.compress_history(session_id, context, n)
        # 归档即抽取：被压缩掉的消息喂给长期记忆抽取器（提炼稳定事实，失败不影响压缩）
        archived_msgs = messages[:n]
        if archived_msgs:
            await _extract_messages(archived_msgs)
    except Exception:
        # 后台压缩失败不影响主流程
        pass


# ---------- 长期记忆自动抽取（write_from_session 触发） ----------


def _schedule_memory_extraction(session_id: str) -> None:
    """把长期记忆抽取调度到后台任务，避免阻塞 SSE done 事件。"""
    task = asyncio.create_task(_run_memory_extraction(session_id))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


async def _extract_messages(messages: list[dict]) -> None:
    """把一批消息交给长期记忆抽取器（内部自动整合 ADD/UPDATE/DELETE）。失败仅打日志。"""
    try:
        if memory_store.is_ready():
            await memory_store.write_from_session(messages)
    except Exception:
        logger.warning("[chat] 记忆抽取失败（不影响响应）", exc_info=True)


async def _run_memory_extraction(session_id: str) -> None:
    """后台记忆抽取：从会话中取出距上次抽取的新消息，写长期记忆。

    游标 memory_extracted_until（会话记录加性字段）标记已抽取的消息条数，
    避免同一批消息反复调用 LLM。无论是否实际写入，都推进游标。
    """
    try:
        session_manager = agent_manager.session_manager
        if session_manager is None or not memory_store.is_ready():
            return
        record = session_manager.get_history(session_id)
        messages = record.get("messages", [])
        if not messages:
            return
        until = int(record.get("memory_extracted_until") or 0)
        batch = messages[until:]
        # 消息数不足不值得一次 LLM 调用，等累计更多再抽
        if len(batch) < MEMORY_EXTRACT_MIN_MESSAGES:
            return
        # 单次最多喂最近 20 条（抽取器输入上限）
        await _extract_messages(batch[-20:])
        # 无论是否写入都推进游标，避免同一批反复调用
        session_manager.set_record_field(session_id, "memory_extracted_until", len(messages))
    except Exception:
        logger.warning("[chat] 记忆抽取调度失败（不影响响应）", exc_info=True)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str
    stream: bool = True


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _new_segment() -> dict[str, Any]:
    return {"content": "", "tool_calls": [], "retrieval_steps": []}


async def _stream_with_tracing(
    message: str,
    history: list[dict[str, Any]],
    session_id: str,
    is_first_user_message: bool,
):
    """在 LangSmith tracing_context 下消费 agent 事件流，为本次请求附加可过滤元数据。

    每次请求作为一个独立 trace 根，可按 session_id 过滤检索链路。
    """
    with tracing_context(
        metadata={
            "session_id": session_id,
            "is_first_user_message": is_first_user_message,
        }
    ):
        async for event in agent_manager.astream(message, history):
            yield event


@router.post("/chat")#接收用户消息，让 Agent 流式回复，同时保存聊天记录。
async def chat(payload: ChatRequest):
    session_manager = agent_manager.session_manager
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")

    history_record = session_manager.load_session_record(payload.session_id)# 加载会话记录
    history = session_manager.load_session_for_agent(payload.session_id)# 加载Agent历史
    #检查是否是第一条用户消息
    is_first_user_message = not any(
        message.get("role") == "user"
        for message in history_record.get("messages", [])
    )
    #产出 SSE 事件，推给前端
    async def event_generator():
        segments: list[dict[str, Any]] = []# 所有段的列表
        current_segment = _new_segment()# 当前正在填充的段
        conversation_saved = False# 是否已存盘（防止重复存）

        # 保存当前段到 segments 列表，并将用户消息和所有段落持久化到磁盘。
        #fallback_content: 错误时使用的兜底内容
        def persist_segments(fallback_content: str | None = None) -> None:
            nonlocal current_segment, conversation_saved
            if conversation_saved:
                return

            if fallback_content:
                if current_segment["content"].strip():
                    current_segment["content"] = (
                        f"{current_segment['content'].rstrip()}\n\n{fallback_content}"
                    )
                else:
                    current_segment["content"] = fallback_content
            #如果有还没存进去的内容则存进去
            if (
                current_segment["content"].strip()
                or current_segment["tool_calls"]
                or current_segment["retrieval_steps"]
            ):
                segments.append(current_segment)
                current_segment = _new_segment()
            #存用户消息
            session_manager.save_message(payload.session_id, "user", payload.message)
            #存AI消息
            for segment in segments:
                session_manager.save_message(
                    payload.session_id,
                    "assistant",
                    segment["content"],
                    tool_calls=segment["tool_calls"] or None,
                    retrieval_steps=segment["retrieval_steps"] or None,
                )

            conversation_saved = True

        try:
            # 是否需要生成会话标题：done 后置位，流耗尽后统一处理
            send_title = False

            #接收agent.py的astream方法返回的流式事件流，并根据事件类型处理不同的内容：token（文本内容）、tool_start/tool_end（工具调用信息）、retrieval（检索步骤）、new_response（新的回复段落开始）和done（回复完成）。在处理过程中，实时将事件通过 SSE 发送给前端，并在完成或发生异常时调用 persist_segments 函数将对话内容保存到磁盘。
            async for event in _stream_with_tracing(
                payload.message, history, payload.session_id, is_first_user_message
            ):
                #event：{"type": "token", "content": "根"}
                event_type = event["type"]

                if event_type == "token":
                    current_segment["content"] += event.get("content", "")
                elif event_type == "tool_start":
                    current_segment["tool_calls"].append(
                        {
                            "tool": event.get("tool", "tool"),
                            "input": event.get("input", ""),
                            "output": "",# 输出先留空，等 tool_end 填
                        }
                    )
                elif event_type == "tool_end":
                    if current_segment["tool_calls"]:
                        current_segment["tool_calls"][-1]["output"] = event.get("output", "")
                elif event_type == "retrieval":
                    current_segment["retrieval_steps"].append(
                        {
                            "kind": event.get("kind", "knowledge"),# 检索类型：memory（长期记忆）/ knowledge（知识库）
                            "stage": event.get("stage", "unknown"), # 检索阶段：vector / bm25 / rerank / memory
                            "title": event.get("title", "检索结果"), # 前端卡片标题，如"向量检索结果（4 路）"
                            "message": event.get("message", ""), # 检索原因或结果说明，如"向量检索已返回补充证据。"
                            "results": event.get("results", []), # 检索到的文档片段列表
                        }
                    )
                elif event_type == "new_response":
                    if (
                        current_segment["content"].strip()
                        or current_segment["tool_calls"]
                        or current_segment["retrieval_steps"]
                    ):
                        segments.append(current_segment)
                    current_segment = _new_segment()
                #如果当前段是空的，就把 done 带的内容补上，防止存一条空消息。
                elif event_type == "done":
                    if not current_segment["content"].strip() and event.get("content"):
                        current_segment["content"] = event["content"]
                    persist_segments()
                    # 自动压缩移出 SSE 尾部：先 persist（当前回合落盘），再调度后台任务，
                    # done 事件立即返回，不阻塞响应结尾。
                    _schedule_compression(payload.session_id)
                    # 长期记忆自动抽取：会话空闲时后台提炼稳定事实（抽取器+整合决策器）
                    _schedule_memory_extraction(payload.session_id)
                #从事件中去掉 type 字段，只保留 SSE 事件需要的字段。
                data = {key: value for key, value in event.items() if key != "type"}
                yield _sse(event_type, data)
                # AI 回复完毕 + 这是该会话的第一条用户消息 → 标记稍后生成标题
                if event_type == "done":
                    send_title = is_first_user_message
            # 标题生成移到流结束后：done 后立即耗尽根生成器（LangSmith 根 run 此刻完成），
            # 避免在 title 的长 await 期间客户端断开导致根 run 被 aclose 标记为 error。
            if send_title:
                title = await agent_manager.generate_title(payload.message)#LLM 生成标题
                session_manager.set_title(payload.session_id, title)#更新会话标题
                yield _sse(
                    "title",
                    {"session_id": payload.session_id, "title": title},
                )
        except Exception as exc:
            persist_segments(fallback_content=f"请求失败: {str(exc) or 'unknown error'}")
            yield _sse("error", {"error": str(exc)})
    #流式响应
    if payload.stream:
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    #非流式响应
    final_text = ""
    async for raw_event in event_generator():
        if raw_event.startswith("event: done"):
            final_text = raw_event
    return JSONResponse({"content": final_text})
