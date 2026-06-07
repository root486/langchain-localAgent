from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv

ENV_FILE_NAME = ".env"

LLM_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "zhipu": {
        "model": "glm-5",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
    },
    "bailian": {
        "model": "qwen3.5-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
    },
    "openai": {
        "model": "gpt-4.1-mini",
        "base_url": "https://api.openai.com/v1",
    },
}

EMBEDDING_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "zhipu": {
        "model": "embedding-3",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
    },
    "bailian": {
        "model": "text-embedding-v4",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "openai": {
        "model": "text-embedding-3-small",
        "base_url": "https://api.openai.com/v1",
    },
}

PROVIDER_ALIASES = {
    "glm": "zhipu",
    "zhipuai": "zhipu",
    "bigmodel": "zhipu",
    "aliyun": "bailian",
    "dashscope": "bailian",
    "qwen": "bailian",
    "openai-compatible": "openai",
    "compatible": "openai",
}


@dataclass(frozen=True)
class Settings:
    backend_dir: Path
    project_root: Path
    llm_provider: str
    llm_model: str
    llm_api_key: str | None
    llm_base_url: str
    embedding_provider: str
    embedding_model: str
    embedding_api_key: str | None
    embedding_base_url: str
    component_char_limit: int = 20_000
    terminal_timeout_seconds: int = 30


def _load_env_file() -> Path:
    backend_dir = Path(__file__).resolve().parent
    env_file_path = _resolve_env_file_path()
    if env_file_path is not None:
        load_dotenv(env_file_path)
    return backend_dir


def _resolve_env_file_path() -> Path | None:
    backend_dir = Path(__file__).resolve().parent
    candidate = backend_dir / ENV_FILE_NAME
    if candidate.exists():
        return candidate
    return None


@lru_cache(maxsize=1)
def _env_file_values() -> dict[str, str]:
    env_file_path = _resolve_env_file_path()
    if env_file_path is None:
        return {}
    values = dotenv_values(env_file_path)
    return {
        key: value.strip()
        for key, value in values.items()
        if isinstance(key, str) and isinstance(value, str) and value.strip()
    }


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _first_config_value(*names: str) -> str | None:
    env_file_values = _env_file_values()
    for name in names:
        value = env_file_values.get(name)
        if value:
            return value
    return _first_env(*names)


def _first_api_key(*names: str) -> str | None:
    return _first_config_value(*names)


def _normalize_provider(
    value: str | None,
    *,
    default: str,
    defaults: dict[str, dict[str, str]],
) -> str:
    normalized = (value or default).strip().lower()
    normalized = PROVIDER_ALIASES.get(normalized, normalized)
    if normalized in defaults:
        return normalized
    return default


def _resolve_llm_api_key(provider: str) -> str | None:
    if provider == "zhipu":
        return _first_api_key("LLM_API_KEY", "ZHIPU_API_KEY", "ZHIPUAI_API_KEY")
    if provider == "bailian":
        return _first_api_key("LLM_API_KEY", "BAILIAN_API_KEY", "DASHSCOPE_API_KEY")
    if provider == "deepseek":
        return _first_api_key("LLM_API_KEY", "DEEPSEEK_API_KEY")
    return _first_api_key("LLM_API_KEY", "OPENAI_API_KEY")


def _resolve_llm_model(provider: str) -> str:
    if provider == "zhipu":
        return _first_config_value("LLM_MODEL", "ZHIPU_MODEL") or LLM_PROVIDER_DEFAULTS[provider]["model"]
    if provider == "bailian":
        return _first_config_value("LLM_MODEL", "BAILIAN_MODEL") or LLM_PROVIDER_DEFAULTS[provider]["model"]
    if provider == "deepseek":
        return _first_config_value("LLM_MODEL", "DEEPSEEK_MODEL") or LLM_PROVIDER_DEFAULTS[provider]["model"]
    return _first_config_value("LLM_MODEL") or LLM_PROVIDER_DEFAULTS[provider]["model"]


def _resolve_llm_base_url(provider: str) -> str:
    if provider == "zhipu":
        return _first_config_value("LLM_BASE_URL", "ZHIPU_BASE_URL") or LLM_PROVIDER_DEFAULTS[provider]["base_url"]
    if provider == "bailian":
        return _first_config_value("LLM_BASE_URL", "BAILIAN_BASE_URL") or LLM_PROVIDER_DEFAULTS[provider]["base_url"]
    if provider == "deepseek":
        return _first_config_value("LLM_BASE_URL", "DEEPSEEK_BASE_URL") or LLM_PROVIDER_DEFAULTS[provider]["base_url"]
    return _first_config_value("LLM_BASE_URL", "OPENAI_BASE_URL") or LLM_PROVIDER_DEFAULTS[provider]["base_url"]


def _resolve_embedding_api_key(provider: str) -> str | None:
    if provider == "zhipu":
        return _first_api_key("EMBEDDING_API_KEY", "ZHIPU_API_KEY", "ZHIPUAI_API_KEY")
    if provider == "bailian":
        return _first_api_key("EMBEDDING_API_KEY", "BAILIAN_API_KEY", "DASHSCOPE_API_KEY")
    return _first_api_key("EMBEDDING_API_KEY", "OPENAI_API_KEY")


def _resolve_embedding_model(provider: str) -> str:
    if provider == "zhipu":
        return (
            _first_config_value("EMBEDDING_MODEL", "ZHIPU_EMBEDDING_MODEL", "ZHIPU_MODEL")
            or EMBEDDING_PROVIDER_DEFAULTS[provider]["model"]
        )
    if provider == "bailian":
        return (
            _first_config_value("EMBEDDING_MODEL", "BAILIAN_EMBEDDING_MODEL", "BAILIAN_MODEL")
            or EMBEDDING_PROVIDER_DEFAULTS[provider]["model"]
        )
    return _first_config_value("EMBEDDING_MODEL") or EMBEDDING_PROVIDER_DEFAULTS[provider]["model"]


def _resolve_embedding_base_url(provider: str) -> str:
    if provider == "zhipu":
        return (
            _first_config_value("EMBEDDING_BASE_URL", "ZHIPU_EMBEDDING_BASE_URL", "ZHIPU_BASE_URL")
            or EMBEDDING_PROVIDER_DEFAULTS[provider]["base_url"]
        )
    if provider == "bailian":
        return (
            _first_config_value("EMBEDDING_BASE_URL", "BAILIAN_EMBEDDING_BASE_URL", "BAILIAN_BASE_URL")
            or EMBEDDING_PROVIDER_DEFAULTS[provider]["base_url"]
        )
    return (
        _first_config_value("EMBEDDING_BASE_URL", "OPENAI_BASE_URL")
        or EMBEDDING_PROVIDER_DEFAULTS[provider]["base_url"]
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    backend_dir = _load_env_file()
    project_root = backend_dir.parent

    llm_provider = _normalize_provider(
        _first_config_value("LLM_PROVIDER"),
        default="zhipu",
        defaults=LLM_PROVIDER_DEFAULTS,
    )
    embedding_provider = _normalize_provider(
        _first_config_value("EMBEDDING_PROVIDER"),
        default="bailian",
        defaults=EMBEDDING_PROVIDER_DEFAULTS,
    )

    return Settings(
        backend_dir=backend_dir,
        project_root=project_root,
        llm_provider=llm_provider,
        llm_model=_resolve_llm_model(llm_provider),
        llm_api_key=_resolve_llm_api_key(llm_provider),
        llm_base_url=_resolve_llm_base_url(llm_provider),
        embedding_provider=embedding_provider,
        embedding_model=_resolve_embedding_model(embedding_provider),
        embedding_api_key=_resolve_embedding_api_key(embedding_provider),
        embedding_base_url=_resolve_embedding_base_url(embedding_provider),
    )


class RuntimeConfigManager:
    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._lock = threading.Lock()
        self._default_config = {"rag_mode": False}

    def load(self) -> dict[str, Any]:
        with self._lock:
            if not self._config_path.exists():
                self.save(self._default_config)
            try:
                return json.loads(self._config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.save(self._default_config)
                return dict(self._default_config)

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        merged = dict(self._default_config)
        merged.update(payload)
        self._config_path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return merged

    def get_rag_mode(self) -> bool:
        return bool(self.load().get("rag_mode", False))

    def set_rag_mode(self, enabled: bool) -> dict[str, Any]:
        return self.save({"rag_mode": enabled})


runtime_config = RuntimeConfigManager(get_settings().backend_dir / "config.json")
