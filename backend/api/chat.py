from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from graph.agent import agent_manager

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str
    stream: bool = True


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _new_segment() -> dict[str, Any]:
    return {"content": "", "tool_calls": [], "retrieval_steps": []}


@router.post("/chat")#接收用户消息，让 Agent 流式回复，同时保存聊天记录。
async def chat(payload: ChatRequest):
    session_manager = agent_manager.session_manager
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")

    history_record = session_manager.load_session_record(payload.session_id)
    history = session_manager.load_session_for_agent(payload.session_id)
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
            
            #接收agent.py的astream方法返回的流式事件流，并根据事件类型处理不同的内容：token（文本内容）、tool_start/tool_end（工具调用信息）、retrieval（检索步骤）、new_response（新的回复段落开始）和done（回复完成）。在处理过程中，实时将事件通过 SSE 发送给前端，并在完成或发生异常时调用 persist_segments 函数将对话内容保存到磁盘。
            async for event in agent_manager.astream(payload.message, history):
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
                            "stage": event.get("stage", "unknown"), # 检索阶段：skill / fallback / vector / bm25 / fused / memory
                            "title": event.get("title", "检索结果"), # 前端卡片标题，如"Skill 检索结果"
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
                #从事件中去掉 type 字段，只保留 SSE 事件需要的字段。
                data = {key: value for key, value in event.items() if key != "type"}
                yield _sse(event_type, data)
                # AI 回复完毕 + 这是该会话的第一条用户消息
                if event_type == "done" and is_first_user_message:
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
