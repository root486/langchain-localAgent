from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat import router as chat_router
from api.config_api import router as config_router
from api.files import router as files_router
from api.knowledge_index import router as knowledge_index_router
from api.sessions import router as sessions_router
from api.tokens import router as tokens_router
from config import get_settings
from graph.agent import agent_manager
from graph.memory_indexer import memory_indexer
from knowledge_retrieval import knowledge_indexer
from tools.skills_scanner import refresh_snapshot

logger = logging.getLogger(__name__)


async def _init_mcp_tools(settings) -> list:
    """初始化 MCP 工具（Tavily 联网搜索等）。启动失败不影响服务，返回空列表。"""
    import os
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_key:
        print("[MCP] TAVILY_API_KEY 未配置，跳过 MCP 工具加载")
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        print("[MCP] langchain-mcp-adapters 未安装，跳过 MCP 工具加载")
        return []

    client = MultiServerMCPClient({
        "tavily": {
            "command": sys.executable,  # 使用当前 Python 解释器，确保能找到 mcp_server_tavily
            "args": ["-m", "mcp_server_tavily"],
            "transport": "stdio",
            "env": {"TAVILY_API_KEY": tavily_key},
        }
    })
    try:
        tools = await client.get_tools()
        print(f"[MCP] 工具加载成功: {[t.name for t in tools]}")
        return tools
    except Exception as exc:
        print(f"[MCP] 工具加载失败（服务不受影响）: {exc}")
        return []


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    refresh_snapshot(settings.backend_dir)
    mcp_tools = await _init_mcp_tools(settings)
    agent_manager.initialize(settings.backend_dir, mcp_tools=mcp_tools)
    memory_indexer.configure(settings.backend_dir)
    memory_indexer.rebuild_index()
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
app.include_router(config_router, prefix="/api", tags=["config"])
app.include_router(knowledge_index_router, prefix="/api", tags=["knowledge"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
