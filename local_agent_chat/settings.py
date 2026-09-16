from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

import yaml

from .installation import config_directory, data_directory


def parse_model(value: str) -> tuple[str, str]:
    provider, separator, model = value.partition(":")
    if provider not in {"openai", "gigachat"} or not separator or not model.strip():
        raise ValueError("Use openai:<model-id> or gigachat:<model-id>.")
    return provider, model


@dataclass(frozen=True)
class ModelProfile:
    id: str
    label: str
    model: str
    api_key_env: str
    api_key: str | None
    base_url: str | None = None
    streaming: bool = True
    summary_options: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        parse_model(self.model)
        if not isinstance(self.summary_options, dict):
            raise ValueError("summary_options must be a mapping of model arguments")
        if {
            "max_tokens",
            "disable_streaming",
            "streaming",
        } & self.summary_options.keys():
            raise ValueError(
                "Summary output and streaming limits are managed by the agent"
            )


@dataclass(frozen=True)
class LLMRetryConfig:
    max_retries: int = 3
    stream_retries: int = 1
    request_timeout_seconds: float = 60.0
    stream_chunk_timeout_seconds: float = 120.0
    auxiliary_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class AgentConfig:
    context_tokens: int = 16000
    summary_trigger_tokens: int = 10000
    keep_tokens: int = 3000
    summary_tokens: int = 1000
    max_output_tokens: int = 2000
    max_model_calls: int = 12

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError("Agent limits must be positive integers")
        if self.summary_tokens * 3 + 2000 >= self.context_tokens:
            raise ValueError(
                "Context must fit summary input, previous summary and output"
            )
        if not self.keep_tokens + self.summary_tokens < self.summary_trigger_tokens:
            raise ValueError("Summary and retained context must fit below the trigger")
        if (
            self.summary_trigger_tokens + self.max_output_tokens + 2000
            > self.context_tokens
        ):
            raise ValueError(
                "Context must reserve 2000 tokens for prompts/tools and room for output"
            )


@dataclass(frozen=True)
class Settings:
    root_path: str
    data_dir: Path
    models: tuple[ModelProfile, ...]
    llm_retry: LLMRetryConfig
    agent: AgentConfig = AgentConfig()


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
    llm_retry = _llm_retry_config()
    directory = config_directory()
    profiles_path = Path(
        os.environ.get("MODEL_PROFILES_FILE") or "models.yaml"
    ).expanduser()
    if not profiles_path.is_absolute():
        profiles_path = directory / profiles_path
    try:
        document = yaml.safe_load(profiles_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {profiles_path}.") from error
    if not isinstance(document, dict) or not isinstance(document.get("models"), list):
        raise ValueError(f"{profiles_path} must contain a models list.")
    identifiers = set()
    for index, item in enumerate(document["models"], 1):
        if not isinstance(item, dict):
            raise ValueError(f"Model profile {index} must be a mapping.")
        for key in ("id", "label", "model", "api_key_env"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Model profile {index} requires a nonempty {key}.")
        if item["id"] in identifiers:
            raise ValueError("Model profile ids must be unique.")
        identifiers.add(item["id"])
        if type(item.get("streaming", True)) is not bool:
            raise ValueError(f"Model profile {index}: streaming must be true or false.")
        if item.get("base_url") is not None and not isinstance(item["base_url"], str):
            raise ValueError(f"Model profile {index}: base_url must be a URL string.")
    profiles = tuple(
        ModelProfile(
            id=item["id"],
            label=item["label"],
            model=item["model"],
            api_key_env=item["api_key_env"],
            api_key=os.environ.get(item["api_key_env"]),
            base_url=item.get("base_url"),
            streaming=item.get("streaming", True),
            summary_options=item.get("summary_options", {}),
        )
        for item in document.get("models", [])
    )
    if not profiles:
        raise ValueError("At least one Model Profile must be configured")

    data_dir = Path(
        os.environ.get("APP_DATA_DIR") or data_directory(directory)
    ).expanduser()
    if not data_dir.is_absolute():
        data_dir = directory / data_dir
    return Settings(
        root_path=_root_path(os.environ.get("APP_ROOT_PATH", "")),
        data_dir=data_dir.resolve(),
        models=profiles,
        llm_retry=llm_retry,
        agent=AgentConfig(
            **{
                name: _bounded_int(
                    f"AGENT_{name.upper()}", default, minimum=1, maximum=1_000_000
                )
                for name, default in vars(AgentConfig()).items()
            }
        ),
    )
