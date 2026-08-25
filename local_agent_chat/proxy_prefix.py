from __future__ import annotations

from typing import Any


class RestoreProxyMethodMiddleware:
    """Restore DELETE requests tunneled through body-hostile HTTP proxies."""

    _HEADER = b"x-proxy-method-override"

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        if scope["type"] == "http" and scope.get("method") == "POST":
            headers = scope.get("headers", [])
            override = next(
                (value for name, value in headers if name.lower() == self._HEADER),
                b"",
            )
            if override.upper() == b"DELETE":
                scope = dict(scope)
                scope["method"] = "DELETE"
                scope["headers"] = [
                    (name, value)
                    for name, value in headers
                    if name.lower() != self._HEADER
                ]
        await self.app(scope, receive, send)


class RestoreProxyPrefixMiddleware:
    """Restore a public prefix removed by Jupyter/VS Code reverse proxies."""

    def __init__(self, app, prefix: str) -> None:
        self.app = app
        self.prefix = prefix.rstrip("/")

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        if self.prefix and scope["type"] in {"http", "websocket"}:
            path = scope.get("path", "") or "/"
            if path != self.prefix and not path.startswith(f"{self.prefix}/"):
                scope = dict(scope)
                scope["path"] = f"{self.prefix}{path}"
                scope["raw_path"] = scope["path"].encode("utf-8")
        await self.app(scope, receive, send)
