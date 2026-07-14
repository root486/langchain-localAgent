from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from cache.redis_client import redis_client

# Redis 会话 TTL（秒）
SESSION_REDIS_TTL = 86400 * 7  # 7 天


class SessionManager:
    """会话管理：Redis 热存储 + JSON 文件冷存储。

    Redis 不可用时自动降级为纯文件模式，不影响现有功能。
    
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.sessions_dir = base_dir / "sessions"
        self.archive_dir = self.sessions_dir / "archive"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Redis helpers ----------

    @staticmethod
    def _redis_key(session_id: str) -> str:
        return f"session:{session_id}"

    def _redis_read(self, session_id: str) -> dict[str, Any] | None:
        raw = redis_client.get(self._redis_key(session_id))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _redis_write(self, record: dict[str, Any]) -> None:
        redis_client.set(
            self._redis_key(str(record["id"])),
            json.dumps(record, ensure_ascii=False),
            ttl=SESSION_REDIS_TTL,
        )

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def _default_record(self, session_id: str, title: str = "新会话") -> dict[str, Any]:
        now = time.time()
        return {
            "id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "compressed_context": "",
            "messages": [],
        }

    def _read_session_file(self, session_id: str) -> dict[str, Any]:
        # 优先从 Redis 热层读取
        cached = self._redis_read(session_id)
        if cached is not None:
            return cached
        # 降级到磁盘文件
        path = self._session_path(session_id)
        if not path.exists():
            record = self._default_record(session_id)
            self._write_session(record)
            return record

        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            record = self._default_record(session_id)
            record["messages"] = raw
            self._write_session(record)
            return record

        raw.setdefault("id", session_id)
        raw.setdefault("title", "新会话")
        raw.setdefault("created_at", time.time())
        raw.setdefault("updated_at", raw["created_at"])
        raw.setdefault("compressed_context", "")
        raw.setdefault("messages", [])
        # 回写到 Redis 热层
        self._redis_write(raw)
        return raw

    def _write_session(self, record: dict[str, Any]) -> None:
        session_id = str(record["id"])
        record["updated_at"] = time.time()
        # 写磁盘（冷存储，永久保留）
        self._session_path(session_id).write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 写 Redis（热存储，7 天 TTL）
        self._redis_write(record)

    def create_session(self, title: str = "新会话") -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        record = self._default_record(session_id, title=title)
        self._write_session(record)
        return record

    #扫描 sessions 文件夹，列出所有会话的摘要信息，按最近更新时间倒序排列。
    def list_sessions(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in self.sessions_dir.glob("*.json"):
            if path.parent == self.archive_dir:
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            records.append(
                {
                    "id": record.get("id", path.stem),
                    "title": record.get("title", "新会话"),
                    "created_at": record.get("created_at"),
                    "updated_at": record.get("updated_at"),
                    "message_count": len(record.get("messages", [])),
                }
            )
        return sorted(records, key=lambda item: item.get("updated_at") or 0, reverse=True)

    def load_session_record(self, session_id: str) -> dict[str, Any]:
        return self._read_session_file(session_id)

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        return self._read_session_file(session_id)["messages"]

    def load_session_for_agent(self, session_id: str) -> list[dict[str, str]]:
        record = self._read_session_file(session_id)
        merged: list[dict[str, str]] = []#存放的是给 Agent 用的对话历史
        #如果压缩了历史消息，压缩后的消息添加到 merged 中
        compressed_context = record.get("compressed_context", "").strip()
        if compressed_context:
            merged.append(
                {
                    "role": "assistant",
                    "content": f"[以下是之前对话的摘要]\n{compressed_context}",
                }
            )
        #合并连续 assistant 消息
        for message in record.get("messages", []):
            role = message.get("role", "")
            content = str(message.get("content", "") or "")
            if role == "assistant" and merged and merged[-1]["role"] == "assistant":
                if content:
                    if merged[-1]["content"]:
                        merged[-1]["content"] += "\n\n" + content  # 上一条不为空则追加
                    else:
                        merged[-1]["content"] = content  # 上一条为空则替换
                continue

            merged.append({"role": role, "content": content})

        return [item for item in merged if item["role"] in {"user", "assistant"}]

    # 保存消息到磁盘
    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        retrieval_steps: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        record = self._read_session_file(session_id)
        message: dict[str, Any] = {"role": role, "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if retrieval_steps:
            message["retrieval_steps"] = retrieval_steps
        record["messages"].append(message)
        self._write_session(record)
        return message

    def get_history(self, session_id: str) -> dict[str, Any]:
        return self._read_session_file(session_id)

    def rename_session(self, session_id: str, title: str) -> dict[str, Any]:
        record = self._read_session_file(session_id)
        record["title"] = title.strip() or "新会话"
        self._write_session(record)
        return record

    def set_title(self, session_id: str, title: str) -> dict[str, Any]:
        return self.rename_session(session_id, title)

    def delete_session(self, session_id: str) -> None:
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
        redis_client.delete(self._redis_key(session_id))

    def compress_history(self, session_id: str, summary: str, n_messages: int) -> dict[str, int]:
        record = self._read_session_file(session_id)
        messages = record.get("messages", [])
        archived = messages[:n_messages] # 前一半要归档
        remaining = messages[n_messages:] # 后一半保留
        # 原始消息归档到磁盘
        archive_path = self.archive_dir / f"{session_id}_{int(time.time())}.json"
        archive_payload = {
            "session_id": session_id,
            "archived_at": time.time(),
            "messages": archived,
        }
        archive_path.write_text(
            json.dumps(archive_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 摘要拼接到 compressed_context
        existing_summary = record.get("compressed_context", "").strip()
        if existing_summary:
            record["compressed_context"] = f"{existing_summary}\n---\n{summary.strip()}"
        else:
            record["compressed_context"] = summary.strip()
        record["messages"] = remaining# 只保留后一半
        self._write_session(record)
        return {
            "archived_count": len(archived),
            "remaining_count": len(remaining),
        }

    def get_compressed_context(self, session_id: str) -> str:
        return self._read_session_file(session_id).get("compressed_context", "")
