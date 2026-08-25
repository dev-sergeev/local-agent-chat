from __future__ import annotations

import re

_LABEL_PREFIX = re.compile(r"^(?:заголовок|название)\s*:\s*", re.IGNORECASE)


def normalize_tool_title(value: str) -> str | None:
    """Turn a model response into a plain three-to-five-word UI label."""

    first_line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    plain = first_line.replace("**", "").replace("__", "").replace("`", "")
    plain = _LABEL_PREFIX.sub("", plain).strip(" \t\"'«»*#.:;!?—–-")
    words = plain.split()
    if len(words) < 3:
        return None
    return " ".join(words[:5]).strip(" \t\"'«»*#.:;!?—–-") or None
