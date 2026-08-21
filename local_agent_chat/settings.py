from __future__ import annotations

import os
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Settings:
    root_path: str
    data_dir: Path
    models: tuple[ModelProfile, ...]


def _root_path(value: str) -> str:
    stripped = value.strip().strip("/")
    return f"/{stripped}" if stripped else ""


def load_settings() -> Settings:
    load_dotenv(override=False)
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
    )
