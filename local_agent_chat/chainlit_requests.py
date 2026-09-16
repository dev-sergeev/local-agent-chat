"""Reject overlapping user requests before Chainlit persists or schedules them."""

from __future__ import annotations

import inspect
from collections.abc import Callable

from chainlit.context import init_ws_context
from chainlit.server import sio
from chainlit.session import WebsocketSession

from .chainlit_data import SQLiteChainlitDataLayer
from .chainlit_revision import sync_chat_history

_ORIGINAL = "_localchat_request_guard_original"


def install_request_guard(
    layer: SQLiteChainlitDataLayer, model_busy: Callable[[], bool]
) -> None:
    """Reserve one user request across every chat until its task finishes."""

    handlers = sio.handlers.get("/", {})
    originals = {}
    for event in ("client_message", "edit_message"):
        handler = handlers.get(event)
        original = getattr(handler, _ORIGINAL, handler)
        if original is None or tuple(inspect.signature(original).parameters) != (
            "sid",
            "payload",
        ):
            raise RuntimeError(f"Unsupported Chainlit {event} signature")
        originals[event] = original

    reserved = False

    def release(_task) -> None:
        nonlocal reserved
        reserved = False

    def guard(event, original):
        async def handle(sid, payload):
            nonlocal reserved
            session = WebsocketSession.require(sid)
            previous = session.current_task
            session_busy = previous is not None and not previous.done()
            if reserved or model_busy() or session_busy:
                ctx = init_ws_context(session)
                # New messages appear optimistically in the browser. Remove
                # only an unpersisted proposal, never an existing history row.
                if event == "client_message" and isinstance(payload, dict):
                    message = payload.get("message")
                    identifier = (
                        message.get("id") if isinstance(message, dict) else None
                    )
                    if isinstance(identifier, str) and not await layer.get_step(
                        identifier
                    ):
                        await ctx.emitter.delete_step({"id": identifier})
                await sync_chat_history(layer)
                await ctx.emitter.send_toast(
                    "Модель занята. Дождитесь завершения текущего запроса и отправьте снова.",
                    "info",
                )
                # Do not disable Stop for a request already running in this tab.
                if not session_busy:
                    await ctx.emitter.task_end()
                return

            # No await between checking availability and claiming the slot.
            reserved = True
            try:
                await original(sid, payload)
                task = session.current_task
                if task is None or task is previous:
                    reserved = False
                else:
                    task.add_done_callback(release)
            except BaseException:
                reserved = False
                raise

        setattr(handle, _ORIGINAL, original)
        return handle

    for event, original in originals.items():
        handlers[event] = guard(event, original)
