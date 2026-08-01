from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat import router as chat_router
from api.files import router as files_router
from api.knowledge_index import router as knowledge_index_router
from api.sessions import router as sessions_router
from api.tokens import router as tokens_router
from config import get_settings
from graph.agent import agent_manager
from graph.memory_store import memory_store
from knowledge_retrieval import knowledge_indexer
from tools.skills_scanner import refresh_snapshot

logger = logging.getLogger(__name__)


async def _load_server_tools(
    client, name: str, *, attempts: int = 3, timeout: float = 30
) -> list:
    """带超时和重试地加载单个 MCP server 的工具，全部失败返回空列表（不影响服务）。"""
    for attempt in range(1, attempts + 1):
        try:
            return await asyncio.wait_for(
                client.get_tools(server_name=name), timeout=timeout
            )
        except Exception as exc:
            print(f"[MCP] {name} 工具加载失败（第 {attempt}/{attempts} 次）: {exc}")
            if attempt < attempts:
                await asyncio.sleep(1)  # 简单退避
    return []


async def _init_mcp_tools(settings) -> list:
    """初始化 MCP 工具（Tavily 联网搜索 / 高德地图等）。启动失败不影响服务，返回空列表。"""
    import os

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        print("[MCP] langchain-mcp-adapters 未安装，跳过 MCP 工具加载")
        return []

    servers: dict = {}

    # Tavily 联网搜索（stdio）
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        servers["tavily"] = {
            "command": sys.executable,  # 使用当前 Python 解释器，确保能找到 mcp_server_tavily
            "args": ["-m", "mcp_server_tavily"],
            "transport": "stdio",
            "env": {"TAVILY_API_KEY": tavily_key},
        }

    # 高德地图（streamable http，key 走 URL 查询参数）
    amap_key = os.getenv("AMAP_API_KEY", "")
    if amap_key:
        servers["amap"] = {
            "transport": "http",  # langchain-mcp-adapters 的 streamable http 传输名
            "url": f"https://mcp.amap.com/mcp?key={amap_key}",
            "timeout": 30,           # 单次 HTTP 请求超时（秒）：API 挂了快速失败，不挂死
            "sse_read_timeout": 60,  # SSE 读流超时（秒）
        }

    if not servers:
        print("[MCP] 未配置 MCP key（TAVILY_API_KEY / AMAP_API_KEY），跳过 MCP 工具加载")
        return []

    client = MultiServerMCPClient(servers)

    # 逐个 server 加载（带超时 + 重试），单个失败不影响其它（如高德端点不可达时 Tavily 仍可用）
    tools = []
    for name in servers:
        server_tools = await _load_server_tools(client, name)
        tools.extend(server_tools)
        if server_tools:
            print(f"[MCP] {name} 工具加载成功: {[t.name for t in server_tools]}")
    return tools


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    # LangSmith 追踪兜底开关：无 Key 或显式关闭（LANGSMITH_TRACING=false）时强制关闭，
    # 避免后台线程反复上传失败刷日志 / 白白消耗免费额度
    if settings.langsmith_api_key and settings.langsmith_enabled:
        os.environ["LANGSMITH_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    else:
        os.environ["LANGSMITH_TRACING_V2"] = "false"
        os.environ.pop("LANGSMITH_API_KEY", None)
    refresh_snapshot(settings.backend_dir)
    mcp_tools = await _init_mcp_tools(settings)
    agent_manager.initialize(settings.backend_dir, mcp_tools=mcp_tools)
    memory_store.configure(settings.backend_dir)
    # 长期记忆遗忘规则：每次启动执行一次（归档长期未用记忆 + 超限淘汰），失败不影响启动
    try:
        forget_result = memory_store.run_forget_rules()
        if forget_result.get("archived") or forget_result.get("pruned"):
            logger.info("[memory_store] 遗忘规则执行: %s", forget_result)
    except Exception:
        logger.warning("[memory_store] 遗忘规则执行失败（不影响启动）", exc_info=True)
    knowledge_indexer.configure(settings.backend_dir)
    if not knowledge_indexer.status().ready:
        knowledge_indexer.rebuild_index()
    yield


app = FastAPI(
    title="Mini-OpenClaw API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(sessions_router, prefix="/api", tags=["sessions"])
app.include_router(files_router, prefix="/api", tags=["files"])
app.include_router(tokens_router, prefix="/api", tags=["tokens"])
app.include_router(knowledge_index_router, prefix="/api", tags=["knowledge"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
