from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator

from langchain.agents import create_agent

from knowledge_retrieval.types import Evidence, SkillRetrievalResult
from tools.python_repl_tool import PythonReplTool
from tools.read_file_tool import ReadFileTool
from tools.terminal_tool import TerminalTool


JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
PATH_PATTERN = re.compile(r"(?:(?:skills|knowledge|workspace|memory|storage|api|graph|tools)/[^\s\"'`]+)")


class SkillRetrieverAgent:
    def __init__(self) -> None:
        self.base_dir = None
        self._model_builder = None

    def configure(self, base_dir, model_builder) -> None:
        self.base_dir = base_dir
        self._model_builder = model_builder

    def _extract_json(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        candidates = [stripped]
        match = JSON_BLOCK_PATTERN.search(stripped)
        if match:
            candidates.insert(0, match.group(1))

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            candidates.append(stripped[start : end + 1])

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return {}

    def _extract_paths(self, tool_calls: list[dict[str, str]]) -> list[str]:
        paths: list[str] = []
        seen: set[str] = set()

        def add_path(value: str) -> None:
            normalized = value.strip().rstrip(".,:;)]}>")
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            paths.append(normalized)

        for tool_call in tool_calls:
            raw_input = str(tool_call.get("input", ""))
            raw_output = str(tool_call.get("output", ""))
            for source in (raw_input, raw_output):
                for match in PATH_PATTERN.findall(source):
                    add_path(match)
            try:
                payload = json.loads(raw_input)
            except json.JSONDecodeError:
                payload = {}
            path_value = payload.get("path")
            if isinstance(path_value, str):
                add_path(path_value)
        return paths

    def _normalize_types(self, value: Any, evidences: list[Evidence], searched_paths: list[str]) -> list[str]:
        candidates: list[str] = []
        if isinstance(value, list):
            candidates.extend(str(item).strip().lower() for item in value if str(item).strip())
        for item in evidences:
            if item.source_type:
                candidates.append(item.source_type.lower())
        for path in searched_paths:
            suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
            if suffix in {"md", "json", "xlsx", "xls", "pdf", "txt"}:
                candidates.append(suffix)

        aliases = {"xlsx": "excel", "xls": "excel"}
        normalized: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            mapped = aliases.get(candidate, candidate)
            if mapped and mapped not in seen:
                seen.add(mapped)
                normalized.append(mapped)
        return normalized

    def _build_result(self, content: str, tool_calls: list[dict[str, str]]) -> SkillRetrievalResult:
        payload = self._extract_json(content)
        raw_evidences = payload.get("evidences", [])
        evidences: list[Evidence] = []
        if isinstance(raw_evidences, list):
            for item in raw_evidences:
                if not isinstance(item, dict):
                    continue
                source_path = str(item.get("source_path", "")).strip()
                if not source_path:
                    continue
                score_value = item.get("score")
                try:
                    score = float(score_value) if score_value is not None else None
                except (TypeError, ValueError):
                    score = None
                raw_parent_id = item.get("parent_id")
                parent_id = str(raw_parent_id).strip() if raw_parent_id else None
                evidences.append(
                    Evidence(
                        source_path=source_path,
                        source_type=str(item.get("source_type", "")).strip().lower() or "unknown",
                        locator=str(item.get("locator", "")).strip(),
                        snippet=str(item.get("snippet", "")).strip(),
                        channel="skill",
                        score=score,
                        parent_id=parent_id,
                    )
                )

        searched_paths = self._extract_paths(tool_calls)
        narrowed_paths: list[str] = []
        raw_narrowed_paths = payload.get("narrowed_paths", [])
        if isinstance(raw_narrowed_paths, list):
            for item in raw_narrowed_paths:
                normalized = str(item).strip()
                if normalized and normalized not in narrowed_paths:
                    narrowed_paths.append(normalized)

        for path in searched_paths:
            if path.startswith("knowledge/") and path not in narrowed_paths:
                narrowed_paths.append(path)

        status = str(payload.get("status", "")).strip().lower()
        if status not in {"success", "partial", "not_found", "uncertain"}:
            status = "success" if evidences else "uncertain"

        reason = str(payload.get("reason", "")).strip()
        if not reason:
            if status == "success":
                reason = "Skill 检索已找到可用于回答的问题证据。"
            elif status == "partial":
                reason = "Skill 检索找到部分证据，但不足以独立完成回答。"
            elif status == "not_found":
                reason = "Skill 检索未在当前知识库范围内找到直接证据。"
            else:
                reason = "Skill 检索结果不确定，可能需要补充召回。"

        rewritten_queries = payload.get("rewritten_queries", [])
        if not isinstance(rewritten_queries, list):
            rewritten_queries = []
        normalized_queries = [str(item).strip() for item in rewritten_queries if str(item).strip()]

        return SkillRetrievalResult(
            status=status,
            evidences=evidences[:5],
            narrowed_paths=narrowed_paths,
            narrowed_types=self._normalize_types(payload.get("narrowed_types"), evidences, searched_paths),
            rewritten_queries=normalized_queries[:5],
            searched_paths=searched_paths,
            reason=reason,
        )

    async def astream(self, query: str) -> AsyncIterator[dict[str, Any]]:
        if self.base_dir is None or self._model_builder is None:
            raise RuntimeError("SkillRetrieverAgent is not configured")

        system_prompt = (
            "You are a local knowledge retrieval agent. "
            "You are not the user-facing assistant and must not answer the user's question directly. "
            "Your first action must be to call read_file on `skills/rag-skill/SKILL.md` and follow that workflow. "
            "Use only local tools to inspect files under `knowledge/`. "
            "Do not use the network. "
            "Return strict JSON only with fields: "
            "status, reason, narrowed_paths, narrowed_types, rewritten_queries, searched_paths, evidences. "
            "Each evidence must contain source_path, source_type, locator, snippet, score, parent_id. "
            "If evidence is insufficient, set status to partial, not_found, or uncertain. "
            "Keep at most 5 evidences."
        )
        tools = [
            TerminalTool(root_dir=self.base_dir),
            PythonReplTool(root_dir=self.base_dir),
            ReadFileTool(root_dir=self.base_dir),
        ]
        agent = create_agent(
            model=self._model_builder(),
            tools=tools,
            system_prompt=system_prompt,
        )

        pending_tools: dict[str, dict[str, str]] = {}
        recorded_tools: list[dict[str, str]] = []
        final_parts: list[str] = []
        last_ai_message = ""

        async for mode, payload in agent.astream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, metadata = payload
                if metadata.get("langgraph_node") != "model":
                    continue
                text = getattr(chunk, "content", "")
                if isinstance(text, str) and text:
                    final_parts.append(text)
                elif isinstance(text, list):
                    for item in text:
                        if isinstance(item, dict) and item.get("type") == "text":
                            final_parts.append(str(item.get("text", "")))
                continue

            if mode != "updates":
                continue

            for update in payload.values():
                for agent_message in update.get("messages", []):
                    message_type = getattr(agent_message, "type", "")
                    tool_calls = getattr(agent_message, "tool_calls", []) or []

                    if message_type == "ai" and not tool_calls:
                        candidate = getattr(agent_message, "content", "")
                        if isinstance(candidate, str) and candidate:
                            last_ai_message = candidate

                    if tool_calls:
                        for tool_call in tool_calls:
                            call_id = str(tool_call.get("id") or tool_call.get("name"))
                            tool_name = str(tool_call.get("name", "tool"))
                            tool_args = tool_call.get("args", "")
                            if not isinstance(tool_args, str):
                                tool_args = json.dumps(tool_args, ensure_ascii=False)
                            pending_tools[call_id] = {
                                "tool": tool_name,
                                "input": str(tool_args),
                            }
                            yield {
                                "type": "tool_start",
                                "tool": tool_name,
                                "input": str(tool_args),
                            }

                    if message_type == "tool":
                        tool_call_id = str(getattr(agent_message, "tool_call_id", ""))
                        pending = pending_tools.pop(
                            tool_call_id,
                            {"tool": getattr(agent_message, "name", "tool"), "input": ""},
                        )
                        output = getattr(agent_message, "content", "")
                        if not isinstance(output, str):
                            output = str(output or "")
                        recorded_tools.append(
                            {
                                "tool": pending["tool"],
                                "input": pending["input"],
                                "output": output,
                            }
                        )
                        yield {
                            "type": "tool_end",
                            "tool": pending["tool"],
                            "output": output,
                        }

        final_content = "".join(final_parts).strip() or last_ai_message.strip()
        yield {
            "type": "skill_result",
            "result": self._build_result(final_content, recorded_tools).to_dict(),
        }


skill_retriever_agent = SkillRetrieverAgent()
