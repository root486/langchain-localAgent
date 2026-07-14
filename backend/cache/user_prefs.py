"""用户偏好 Redis 存储。对比原来 USER.md 静态文件，支持 Agent 对话中动态更新，
不需要手动编辑文件、不触发索引重建。

Redis Hash: user:prefs → {key: value, ...}
"""
from __future__ import annotations

import logging
from typing import Any

from cache.redis_client import redis_client

logger = logging.getLogger(__name__)

PREFS_KEY = "user:prefs"

# 默认偏好
DEFAULT_PREFS = {
    "language": "zh",
    "prefers_explainable": "true",
    "prefers_local_first": "true",
}


def _ensure_defaults() -> None:
    """首次使用时写入默认值。"""
    if redis_client.client is None:
        return
    if not redis_client.client.exists(PREFS_KEY):
        redis_client.client.hset(PREFS_KEY, mapping=DEFAULT_PREFS)
        logger.info("用户偏好已初始化（默认值）")


def get_all() -> dict[str, str]:
    """读取全部偏好。Redis 不可用时返回空字典。"""
    if redis_client.client is None:
        return {}
    try:
        _ensure_defaults()
        raw = redis_client.client.hgetall(PREFS_KEY)
        logger.info("用户偏好读取成功: %s", dict(raw))
        return {k: v for k, v in raw.items()}
    except Exception as exc:
        logger.warning("用户偏好读取失败: %s", exc)
        return {}


def set_pref(key: str, value: str) -> bool:
    """写入单条偏好。"""
    if redis_client.client is None:
        return False
    try:
        redis_client.client.hset(PREFS_KEY, key, value)
        logger.info("用户偏好已更新: %s = %s", key, value)
        return True
    except Exception as exc:
        logger.warning("用户偏好写入失败: %s", exc)
        return False


def to_prompt_text(prefs: dict[str, str]) -> str:
    """将偏好字典转成注入 system prompt 的一句话。"""
    if not prefs:
        return ""
    lang = "中文" if prefs.get("language") == "zh" else "中文"
    parts = [f"用户偏好：使用{lang}交流"]
    if prefs.get("prefers_explainable") == "true":
        parts.append("希望系统行为可解释")
    if prefs.get("prefers_local_first") == "true":
        parts.append("优先本地处理")
    # 动态偏好（Agent 自动写入的）
    for key, value in prefs.items():
        if key not in {"language", "prefers_explainable", "prefers_local_first"}:
            parts.append(f"{key}={value}")
    prompt = "。".join(parts) + "。"
    prompt += " 你可以使用 python_repl 执行 from cache.user_prefs import set_pref; set_pref('key', 'value') 来记录新发现的用户偏好。"
    return prompt
