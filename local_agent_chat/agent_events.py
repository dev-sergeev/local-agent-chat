from __future__ import annotations

import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolStarted:
    id: str
    name: str
    input: str


@dataclass(frozen=True)
class ToolFinished:
    id: str
    output: str


@dataclass(frozen=True)
class ToolFailed:
    id: str
    error: str


type AgentEvent = TextDelta | ToolStarted | ToolFinished | ToolFailed
type EventSink = Callable[[AgentEvent], Awaitable[None]]

_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|authorization|access[_-]?token|secret|password|credential)",
    re.IGNORECASE,
)
_ASSIGNMENT_SECRET = re.compile(
    r"(?im)(\b[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)\b\s*[:=]\s*)"
    r"([^\s,;]+|\"[^\"]*\"|'[^']*')"
)
_BEARER_SECRET = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+")
_COMMON_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")


def _redact_structure(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if _SENSITIVE_KEY.search(str(key))
                else _redact_structure(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact_structure(item) for item in value]
    if isinstance(value, bytes):
        return "<binary data>"
    return value


def redact_text(text: str) -> str:
    """Remove common credentials without exposing the application's env values."""

    redacted = _ASSIGNMENT_SECRET.sub(r"\1<redacted>", text)
    redacted = _BEARER_SECRET.sub(r"\1<redacted>", redacted)
    redacted = _COMMON_KEY.sub("<redacted>", redacted)
    for name, value in os.environ.items():
        if _SENSITIVE_KEY.search(name) and len(value) >= 12:
            redacted = redacted.replace(value, "<redacted>")
    return redacted


def safe_text(value: Any, *, max_chars: int = 6000) -> str:
    """Serialize, redact and bound data before it crosses the UI seam."""

    cleaned = _redact_structure(value)
    if isinstance(cleaned, str):
        text = cleaned
    else:
        try:
            text = json.dumps(cleaned, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            text = str(cleaned)
    text = redact_text(text)
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    head = max_chars * 2 // 3
    tail = max_chars - head
    return f"{text[:head]}\n\n… пропущено {omitted} символов …\n\n{text[-tail:]}"


def public_text(content: Any) -> str:
    """Extract only provider-visible answer text, excluding reasoning blocks."""

    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, Mapping):
            continue
        if block.get("type") not in {"text", "output_text"}:
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
        elif isinstance(text, Mapping) and isinstance(text.get("value"), str):
            parts.append(text["value"])
    return "".join(parts)


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    text = public_text(content)
    return text if text else safe_text(content)
