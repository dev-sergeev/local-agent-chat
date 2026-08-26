from __future__ import annotations

import re

DEFAULT_CHAT_TITLE = "Новый диалог"
CHAT_TITLE_STATE_KEY = "chat_title_state"
CHAT_TITLE_PENDING = "pending"
CHAT_TITLE_GENERATED = "generated"
CHAT_TITLE_FALLBACK = "fallback"
CHAT_TITLE_MANUAL = "manual"

_LABEL_PREFIX = re.compile(r"^(?:заголовок|название)\s*:\s*", re.IGNORECASE)
_TITLE_TRIM = " \t\"'«»*#.:;!?—–-"


def chat_title_source(request_text: str, filenames: list[str] | tuple[str, ...]) -> str:
    """Add bounded attachment context when a request has little or no text."""

    text = " ".join(request_text.split())
    safe_names: list[str] = []
    for filename in filenames[:5]:
        basename = str(filename).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        normalized = " ".join(basename.split())[:80].strip()
        if normalized:
            safe_names.append(normalized)
    if not safe_names:
        return text
    attachment_context = f"Файлы: {', '.join(safe_names)}"
    return f"{text}\n{attachment_context}" if text else attachment_context


def normalize_chat_title(value: str) -> str | None:
    """Turn a model response into a compact plain-text Chat title."""

    first_line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    plain = first_line.replace("**", "").replace("__", "").replace("`", "")
    plain = _LABEL_PREFIX.sub("", plain).strip(_TITLE_TRIM)
    words = plain.split()
    if len(words) < 3:
        return None

    selected: list[str] = []
    for word in words[:5]:
        candidate = " ".join((*selected, word))
        if len(candidate) > 64:
            break
        selected.append(word)
    if len(selected) < 3:
        return None
    return " ".join(selected).strip(_TITLE_TRIM) or None


def fallback_chat_title(request_text: str) -> str:
    """Derive a compact deterministic title when the title model is unavailable."""

    plain = request_text.replace("**", "").replace("__", "").replace("`", "")
    plain = _LABEL_PREFIX.sub("", " ".join(plain.split())).strip(_TITLE_TRIM)
    if not plain:
        return DEFAULT_CHAT_TITLE

    selected: list[str] = []
    for word in plain.split()[:5]:
        candidate = " ".join((*selected, word))
        if len(candidate) > 64:
            break
        selected.append(word)
    if selected:
        return " ".join(selected).strip(_TITLE_TRIM) or DEFAULT_CHAT_TITLE
    return f"{plain[:63].rstrip(_TITLE_TRIM)}…"
