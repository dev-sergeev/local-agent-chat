"""One bounded backoff schedule for provider requests and empty streams."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from math import isfinite


def retry_delay(attempt: int, headers: Mapping[str, str] | None = None) -> float:
    """Wait 1, 2, 4, ... seconds, honoring longer Retry-After up to 300s."""

    delay = min(300.0, 2.0 ** min(attempt, 9))
    headers = {key.lower(): value for key, value in (headers or {}).items()}
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            requested = float(value) * scale
        except ValueError:
            if name != "retry-after":
                continue
            try:
                date = parsedate_to_datetime(value)
                requested = (date - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                continue
        if isfinite(requested) and requested >= 0:
            return min(300.0, max(delay, requested))
    return delay
