from __future__ import annotations

import re

_ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_BACKTICK_RUN = re.compile(r"`+")


def _clean(value: str) -> str:
    text = _ANSI_ESCAPE.sub("", value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(
        character
        for character in text
        if character in {"\n", "\t"} or ord(character) >= 32
    ).strip("\n")


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    head_size = limit * 2 // 3
    tail_size = limit - head_size
    omitted = len(value) - limit
    return (
        f"{value[:head_size].rstrip()}\n\n"
        f"… пропущено {omitted} символов …\n\n"
        f"{value[-tail_size:].lstrip()}"
    )


def format_tool_log(value: str, *, limit: int) -> str:
    """Render untrusted tool output as a bounded, Markdown-safe text log."""

    text = _truncate(_clean(value), limit)
    if not text:
        text = "(нет вывода)"
    longest_run = max(
        (len(match.group()) for match in _BACKTICK_RUN.finditer(text)),
        default=0,
    )
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}text\n{text}\n{fence}"
