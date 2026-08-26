from __future__ import annotations

import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class ModelProfile:
    id: str
    label: str
    model: str
    api_key_env: str
    api_key: str | None
    base_url: str | None = None
    streaming: bool = True


@dataclass(frozen=True)
class LLMRetryConfig:
    max_retries: int = 3
    stream_retries: int = 1
    request_timeout_seconds: float = 60.0
    stream_chunk_timeout_seconds: float = 120.0
    auxiliary_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class Settings:
    root_path: str
    data_dir: Path
    models: tuple[ModelProfile, ...]
    llm_retry: LLMRetryConfig


def _root_path(value: str) -> str:
    stripped = value.strip().strip("/")
    return f"/{stripped}" if stripped else ""


def _bounded_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"{name} must be an integer from {minimum} to {maximum}"
        ) from error
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _positive_finite_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a finite number greater than zero") from error
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return value


def _llm_retry_config() -> LLMRetryConfig:
    defaults = LLMRetryConfig()
    return LLMRetryConfig(
        max_retries=_bounded_int(
            "LLM_MAX_RETRIES",
            defaults.max_retries,
            minimum=0,
            maximum=10,
        ),
        stream_retries=_bounded_int(
            "LLM_STREAM_RETRIES",
            defaults.stream_retries,
            minimum=0,
            maximum=10,
        ),
        request_timeout_seconds=_positive_finite_float(
            "LLM_REQUEST_TIMEOUT_SECONDS", defaults.request_timeout_seconds
        ),
        stream_chunk_timeout_seconds=_positive_finite_float(
            "LLM_STREAM_CHUNK_TIMEOUT_SECONDS",
            defaults.stream_chunk_timeout_seconds,
        ),
        auxiliary_timeout_seconds=_positive_finite_float(
            "LLM_AUXILIARY_TIMEOUT_SECONDS", defaults.auxiliary_timeout_seconds
        ),
    )


def load_settings() -> Settings:
    load_dotenv(override=False)
    llm_retry = _llm_retry_config()
    profiles_path = Path(os.environ.get("MODEL_PROFILES_FILE", "models.yaml"))
    document = yaml.safe_load(profiles_path.read_text(encoding="utf-8")) or {}
    profiles = tuple(
        ModelProfile(
            id=item["id"],
            label=item["label"],
            model=item["model"],
            api_key_env=item["api_key_env"],
            api_key=os.environ.get(item["api_key_env"]),
            base_url=item.get("base_url"),
            streaming=item.get("streaming", True),
        )
        for item in document.get("models", [])
    )
    if not profiles:
        raise ValueError("At least one Model Profile must be configured")

    data_dir = Path(os.environ.get("APP_DATA_DIR", ".local-agent-chat")).resolve()
    return Settings(
        root_path=_root_path(os.environ.get("APP_ROOT_PATH", "")),
        data_dir=data_dir,
        models=profiles,
        llm_retry=llm_retry,
    )
