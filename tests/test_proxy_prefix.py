import json

import pytest

from local_agent_chat.proxy_prefix import (
    RestoreProxyMethodMiddleware,
    RestoreProxyPrefixMiddleware,
)


async def _invoke(middleware, scope, body=b""):
    captured = {}
    received = False

    async def app(inner_scope, receive, _send):
        captured["scope"] = inner_scope
        captured["message"] = await receive()

    async def receive():
        nonlocal received
        assert not received
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    await middleware(app)(scope, receive, lambda _message: None)
    return captured


@pytest.mark.asyncio
async def test_restores_public_prefix_removed_by_proxy() -> None:
    captured = await _invoke(
        lambda app: RestoreProxyPrefixMiddleware(app, "/user/test/proxy/8765"),
        {"type": "http", "path": "/project/thread", "headers": []},
    )

    assert captured["scope"]["path"] == ("/user/test/proxy/8765/project/thread")


@pytest.mark.asyncio
async def test_restores_delete_tunneled_as_post_without_changing_its_body() -> None:
    body = json.dumps({"threadId": "chat-1"}).encode()
    captured = await _invoke(
        RestoreProxyMethodMiddleware,
        {
            "type": "http",
            "method": "POST",
            "path": "/user/test/proxy/8765/project/thread",
            "headers": [
                (b"content-type", b"application/json"),
                (b"x-proxy-method-override", b"DELETE"),
            ],
        },
        body,
    )

    assert captured["scope"]["method"] == "DELETE"
    assert (b"x-proxy-method-override", b"DELETE") not in captured["scope"]["headers"]
    assert captured["message"]["body"] == body


@pytest.mark.asyncio
async def test_does_not_override_post_without_the_narrow_proxy_header() -> None:
    captured = await _invoke(
        RestoreProxyMethodMiddleware,
        {
            "type": "http",
            "method": "POST",
            "path": "/project/thread",
            "headers": [(b"content-type", b"application/json")],
        },
    )

    assert captured["scope"]["method"] == "POST"
