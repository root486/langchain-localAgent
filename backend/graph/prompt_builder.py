from __future__ import annotations

from pathlib import Path

from config import get_settings

SYSTEM_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("Skills Snapshot", "SKILLS_SNAPSHOT.md"), # 技能快照
    ("Soul", "workspace/SOUL.md"),#人格
    ("Identity", "workspace/IDENTITY.md"),#身份
)
# 长期记忆说明：MEMORY.md 静态注入已废弃，改为 ChromaDB 简单长期记忆动态检索（见 graph/memory_store.py）
LONG_TERM_MEMORY_NOTE = (
    "<!-- Long-term Memory -->\n"
    "相关长期记忆（用户画像 / 项目事实）将通过检索证据动态注入。"
    "仅依赖当次注入的 MEMORY 片段，不要假设未检索到的记忆仍然有效。"
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


def build_system_prompt(base_dir: Path) -> str:
    settings = get_settings()
    parts: list[str] = []

    for label, relative_path in SYSTEM_COMPONENTS:
        content = _read_component(base_dir, relative_path, settings.component_char_limit)
        parts.append(f"<!-- {label} -->\n{content}")

    # 长期记忆动态检索说明（记忆检索常开，无开关）
    parts.append(LONG_TERM_MEMORY_NOTE)
    parts.append(RUNTIME_OVERRIDE)
    return "\n\n".join(parts)
