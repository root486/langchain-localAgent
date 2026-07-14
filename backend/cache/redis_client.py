"""Redis 客户端封装。

如果 Redis 不可用（未配置或连接失败），自动降级为 no-op 模式，不影响现有功能。
"""
from __future__ import annotations

import logging
from typing import Any

from config import get_settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis 单例客户端，连接失败自动降级。"""

    def __init__(self) -> None:
        self._client: Any = None
        self._available: bool | None = None  # None = 未检测, True/False = 可用/不可用

    def _connect(self) -> Any | None:
        """尝试连接 Redis，失败返回 None。"""
        settings = get_settings()
        if not settings.redis_url:
            return None
        try:
            import redis
            client = redis.Redis.from_url(
                settings.redis_url,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            client.ping()
            return client
        except Exception:
            return None

    @property
    def client(self) -> Any | None:
        """获取 Redis 连接（懒加载 + 缓存检测结果）。"""
        if self._available is None:
            self._client = self._connect()
            self._available = self._client is not None
            if self._available:
                logger.info("Redis 已连接")
            else:
                logger.info("Redis 未配置或不可用，使用降级模式")
        return self._client

    # ---------- 对外 API ----------

    def get(self, key: str) -> str | None:
        if self.client is None:
            return None
        try:
            return self.client.get(key)
        except Exception:
            return None

    def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        if self.client is None:
            return False
        try:
            self.client.set(key, value, ex=ttl)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        if self.client is None:
            return False
        try:
            self.client.delete(key)
            return True
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        if self.client is None:
            return False
        try:
            return bool(self.client.exists(key))
        except Exception:
            return False


# 模块级单例
redis_client = RedisClient()
