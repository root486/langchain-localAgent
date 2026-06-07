from __future__ import annotations

import asyncio

from fastapi import APIRouter

from knowledge_retrieval import knowledge_indexer

router = APIRouter()


@router.get("/knowledge/index/status")
async def get_index_status() -> dict:
    return knowledge_indexer.status().to_dict()


@router.post("/knowledge/index/rebuild")
async def rebuild_index() -> dict[str, bool]:
    if knowledge_indexer.is_building():
        return {"accepted": True}
    asyncio.create_task(asyncio.to_thread(knowledge_indexer.rebuild_index))
    return {"accepted": True}
