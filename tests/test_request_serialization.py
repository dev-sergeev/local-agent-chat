import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import chainlit.socket as chainlit_socket
import pytest
from chainlit.server import sio

from local_agent_chat import chainlit_requests
from local_agent_chat.llm_retry import RetryBlock
from local_agent_chat.settings import LLMRetryConfig


async def test_inference_gate_serializes_main_summary_and_title_and_releases_on_cancel():
    block = RetryBlock(LLMRetryConfig(auxiliary_timeout_seconds=1))
    started = asyncio.Event()
    calls = []

    async def main():
        calls.append("main")
        started.set()
        await asyncio.Event().wait()

    async def summary():
        calls.append("summary")

    async def title():
        calls.append("title")
        return "title"

    first = asyncio.create_task(block.run_streaming_model(main))
    await started.wait()
    assert block.busy
    second = asyncio.create_task(block.run_streaming_model(summary))
    third = asyncio.create_task(block.run_auxiliary(title))
    await asyncio.sleep(0)
    assert calls == ["main"]
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await asyncio.wait_for(third, 1) == "title"
    assert calls == ["main", "title"]
    assert not block.busy


@pytest.mark.parametrize("event", ["client_message", "edit_message"])
@pytest.mark.parametrize("same_session", [False, True])
async def test_socket_rejects_second_request_without_persisting_or_replacing_stop_task(
    monkeypatch, event, same_session
):
    sessions = {
        name: SimpleNamespace(current_task=None) for name in ("first", "second")
    }
    started, release = asyncio.Event(), asyncio.Event()
    accepted = []
    emitter = SimpleNamespace(
        send_toast=AsyncMock(), delete_step=AsyncMock(), task_end=AsyncMock()
    )
    layer = SimpleNamespace(get_step=AsyncMock(return_value=None))

    async def process(session, payload):
        accepted.append(payload)
        started.set()
        await release.wait()

    async def edit(sid, payload):
        session = sessions[sid]
        session.current_task = asyncio.create_task(process(session, payload))

    monkeypatch.setitem(sio.handlers["/"], "client_message", chainlit_socket.message)
    monkeypatch.setitem(sio.handlers["/"], "edit_message", edit)
    monkeypatch.setattr(chainlit_socket, "process_message", process)
    monkeypatch.setattr(
        chainlit_requests.WebsocketSession, "require", sessions.__getitem__
    )
    monkeypatch.setattr(
        chainlit_requests,
        "init_ws_context",
        lambda _s: SimpleNamespace(emitter=emitter),
    )
    monkeypatch.setattr(chainlit_requests, "sync_chat_history", AsyncMock())
    chainlit_requests.install_request_guard(layer, lambda: False)
    await sio.handlers["/"]["client_message"]("first", {"message": {"id": "one"}})
    original_task = sessions["first"].current_task
    await started.wait()
    target = "first" if same_session else "second"
    try:
        await sio.handlers["/"][event](target, {"message": {"id": "two"}})
        assert len(accepted) == 1
        assert sessions["first"].current_task is original_task
        assert sessions["second"].current_task is None
        assert emitter.send_toast.await_count == 1
        assert emitter.task_end.await_count == (0 if same_session else 1)
        assert emitter.delete_step.await_count == (
            1 if event == "client_message" else 0
        )
    finally:
        release.set()
        await original_task
        await asyncio.sleep(0)
    await sio.handlers["/"][event]("second", {"message": {"id": "three"}})
    await sessions["second"].current_task
    assert len(accepted) == 2


async def test_busy_title_rejects_new_user_request(monkeypatch):
    session = SimpleNamespace(current_task=None)
    emitter = SimpleNamespace(
        send_toast=AsyncMock(), delete_step=AsyncMock(), task_end=AsyncMock()
    )
    accepted = []

    async def handler(sid, payload):
        accepted.append(payload)

    for event in ("client_message", "edit_message"):
        monkeypatch.setitem(sio.handlers["/"], event, handler)
    monkeypatch.setattr(
        chainlit_requests.WebsocketSession, "require", lambda _: session
    )
    monkeypatch.setattr(
        chainlit_requests, "init_ws_context", lambda _: SimpleNamespace(emitter=emitter)
    )
    monkeypatch.setattr(chainlit_requests, "sync_chat_history", AsyncMock())
    chainlit_requests.install_request_guard(
        SimpleNamespace(get_step=AsyncMock(return_value=None)), lambda: True
    )
    await sio.handlers["/"]["client_message"]("socket", {"message": {"id": "new"}})
    assert accepted == []
    assert session.current_task is None
    emitter.delete_step.assert_awaited_once_with({"id": "new"})
    emitter.send_toast.assert_awaited_once()
