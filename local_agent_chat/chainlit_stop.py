from __future__ import annotations

import inspect
from functools import wraps

import chainlit.socket as chainlit_socket
from chainlit.config import config
from chainlit.server import sio

_ORIGINAL_HANDLER = "_local_agent_chat_original_stop_handler"
_EXPECTED_PARAMETERS = ("sid",)


def install_localized_stop_compatibility() -> None:
    """Keep Chainlit stop semantics without its hard-coded English message."""

    handlers = sio.handlers.get("/")
    current_handler = handlers.get("stop") if handlers is not None else None
    if current_handler is None:
        raise RuntimeError("Unsupported Chainlit stop handler registration")
    if hasattr(current_handler, _ORIGINAL_HANDLER):
        return
    if tuple(inspect.signature(current_handler).parameters) != _EXPECTED_PARAMETERS:
        raise RuntimeError("Unsupported Chainlit stop signature")

    @wraps(current_handler)
    async def localized_stop(sid: str) -> None:
        if session := chainlit_socket.WebsocketSession.get(sid):
            chainlit_socket.init_ws_context(session)

            if session.current_task:
                session.current_task.cancel()

            if config.code.on_stop:
                await config.code.on_stop()

    setattr(localized_stop, _ORIGINAL_HANDLER, current_handler)
    handlers["stop"] = localized_stop
