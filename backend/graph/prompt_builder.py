from __future__ import annotations

from pathlib import Path

from config import get_settings

SYSTEM_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("Skills Snapshot", "SKILLS_SNAPSHOT.md"), # 技能快照
    ("Soul", "workspace/SOUL.md"),#人格
    ("Identity", "workspace/IDENTITY.md"),#身份
    ("Long-term Memory", "memory/MEMORY.md"),#长期记忆
)
#有检索证据时优先用。
RUNTIME_OVERRIDE = """<!-- Runtime Override -->
When explicit retrieval evidence is provided for the current request, prioritize that evidence.
Do not assume missing evidence exists elsewhere.
For web search or fetching current information, use the tavily_web_search tool directly.
Do NOT write Python code or shell commands for web search.
"""

#截断文本，防止文件太长
def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _read_component(base_dir: Path, relative_path: str, limit: int) -> str:
    path = base_dir / relative_path
    if not path.exists():
        return f"[missing component: {relative_path}]"
    return _truncate(path.read_text(encoding="utf-8"), limit)


def build_system_prompt(base_dir: Path, rag_mode: bool) -> str:
    settings = get_settings()
    parts: list[str] = []

    for label, relative_path in SYSTEM_COMPONENTS:
        #如果是长期记忆文件且开启了 RAG，不读文件内容，而是插入一段提示——告诉 LLM "记忆会动态注入，别自己猜"。
        if rag_mode and relative_path == "memory/MEMORY.md":
            parts.append(
                "<!-- Long-term Memory -->\n"
                "长期记忆将通过检索动态注入。你应优先使用当次检索到的 MEMORY 片段，"
                "不要假设未检索到的记忆仍然有效。"
            )
            continue

        content = _read_component(base_dir, relative_path, settings.component_char_limit)
        parts.append(f"<!-- {label} -->\n{content}")

    # Redis 用户偏好（动态更新，不依赖索引重建）
    from cache.user_prefs import get_all, to_prompt_text
    prefs = get_all()
    if prefs:
        prefs_text = to_prompt_text(prefs)
        parts.append(f"<!-- User Preferences -->\n{prefs_text}")

    parts.append(RUNTIME_OVERRIDE)
    return "\n\n".join(parts)
