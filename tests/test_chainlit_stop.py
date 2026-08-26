from __future__ import annotations

import chainlit.socket as chainlit_socket
import pytest
from chainlit.config import config
from chainlit.server import sio

from local_agent_chat import chainlit_stop, chainlit_ui
from local_agent_chat.chainlit_ui import ChainlitTurnView


@pytest.mark.asyncio
async def test_stop_keeps_cancellation_and_callback_without_english_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent_messages: list[str] = []
    initialized_sessions: list[object] = []
    cancelled_tasks: list[object] = []
    stop_callbacks: list[None] = []
    localized_messages: list[object] = []

    class FakeRoot:
        end = None
        output = ""
        auto_collapse = True
        updates = 0

        async def update(self) -> None:
            self.updates += 1

    class LocalizedMessage:
        def __init__(self, content: str, **kwargs: object) -> None:
            self.content = content
            self.metadata = kwargs.get("metadata", {})

        async def send(self) -> None:
            localized_messages.append(self)

    class FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

        async def send(self) -> None:
            sent_messages.append(self.content)

    class FakeTask:
        def cancel(self) -> None:
            cancelled_tasks.append(self)

    class FakeSession:
        current_task = FakeTask()

    session = FakeSession()
    view = ChainlitTurnView()
    view.root = FakeRoot()

    def init_context(active_session: object) -> None:
        initialized_sessions.append(active_session)

    async def on_stop() -> None:
        stop_callbacks.append(None)
        await view.cancel()

    # Start from Chainlit's handler so the test exercises the compatibility seam
    # used by the running application rather than a synthetic helper function.
    monkeypatch.setitem(sio.handlers["/"], "stop", chainlit_socket.stop)
    monkeypatch.setattr(chainlit_socket.WebsocketSession, "get", lambda _sid: session)
    monkeypatch.setattr(chainlit_socket, "init_ws_context", init_context)
    monkeypatch.setattr(chainlit_socket, "Message", FakeMessage)
    monkeypatch.setattr(chainlit_ui.cl, "Message", LocalizedMessage)
    monkeypatch.setattr(chainlit_ui, "utc_now", lambda: "now")
    monkeypatch.setattr(config.code, "on_stop", on_stop)

    chainlit_stop.install_localized_stop_compatibility()
    installed_handler = sio.handlers["/"]["stop"]
    chainlit_stop.install_localized_stop_compatibility()

    assert sio.handlers["/"]["stop"] is installed_handler
    await installed_handler("socket-1")

    assert initialized_sessions == [session]
    assert cancelled_tasks == [session.current_task]
    assert stop_callbacks == [None]
    assert sent_messages == []
    assert view.root.output == "Остановлено пользователем"
    assert view.root.updates == 1
    assert [message.content for message in localized_messages] == [
        "_Выполнение остановлено пользователем._"
    ]
    assert localized_messages[0].metadata["event_kind"] == "assistant_cancelled"


def test_stop_adapter_rejects_an_unknown_chainlit_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def incompatible_stop(sid: str, reason: str) -> None:
        del sid, reason

    monkeypatch.setitem(sio.handlers["/"], "stop", incompatible_stop)

    with pytest.raises(RuntimeError, match="Unsupported Chainlit stop signature"):
        chainlit_stop.install_localized_stop_compatibility()
