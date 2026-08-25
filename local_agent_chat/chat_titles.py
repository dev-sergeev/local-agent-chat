from __future__ import annotations

import re

DEFAULT_CHAT_TITLE = "Новый диалог"
CHAT_TITLE_STATE_KEY = "chat_title_state"
CHAT_TITLE_PENDING = "pending"
CHAT_TITLE_GENERATED = "generated"
CHAT_TITLE_FALLBACK = "fallback"
CHAT_TITLE_MANUAL = "manual"

_LABEL_PREFIX = re.compile(r"^(?:заголовок|название)\s*:\s*", re.IGNORECASE)


def normalize_chat_title(value: str) -> str | None:
    """Turn a model response into a compact plain-text Chat title."""

    first_line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    plain = first_line.replace("**", "").replace("__", "").replace("`", "")
    plain = _LABEL_PREFIX.sub("", plain).strip(" \t\"'«»*#.:;!?—–-")
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
    return " ".join(selected).strip(" \t\"'«»*#.:;!?—–-") or None
