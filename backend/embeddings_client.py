"""OpenAI 兼容 Embedding 客户端（httpx 直连，替代 LlamaIndex 的 OpenAIEmbedding）。

- 只依赖 httpx（requirements.txt 已有），不引入新依赖
- 兼容 百炼(DashScope) / Zhipu / OpenAI 的 /embeddings 接口
- 批大小固定 10（百炼 text-embedding-v4 单次请求上限），自动分批
"""
from __future__ import annotations

import httpx

from config import get_settings

# 百炼 text-embedding-v4 单次请求最大 10 条
EMBED_BATCH_SIZE = 10
# 单次嵌入请求超时（秒）
EMBED_TIMEOUT_SECONDS = 60.0


class EmbeddingClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.embedding_api_key or ""
        self.base_url = settings.embedding_base_url.rstrip("/")
        self.model = settings.embedding_model
        self.batch_size = EMBED_BATCH_SIZE

    def _ensure_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError("embedding_api_key 未配置，无法调用 embedding API")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文本，自动按 batch_size 分批，返回与输入顺序一致的向量列表。"""
        self._ensure_api_key()
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            resp = httpx.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": batch},
                timeout=EMBED_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            # 按 index 排序，保证与 input 顺序一致
            data.sort(key=lambda d: d.get("index", 0))
            embeddings.extend(d["embedding"] for d in data)
        return embeddings

    def embed_one(self, text: str) -> list[float]:
        """嵌入单条文本（如检索时的 query）。"""
        return self.embed([text])[0]
