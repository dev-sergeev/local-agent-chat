from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable

from chainlit.chat_context import chat_context
from chainlit.context import context, init_ws_context
from chainlit.message import Message
from chainlit.server import sio
from chainlit.session import WebsocketSession

from .chainlit_data import SQLiteChainlitDataLayer

logger = logging.getLogger(__name__)


async def sync_chat_history(layer: SQLiteChainlitDataLayer) -> None:
    """Publish the authoritative timeline, including recovery after a failed edit."""
    thread = await layer.get_thread(context.session.thread_id)
    if thread is None:
        return
    chat_context.clear()
    for step in thread.get("steps", []):
        if step["type"] in {"user_message", "assistant_message", "system_message"}:
            chat_context.add(Message.from_dict(step))
    await context.emitter.resume_thread(thread)


def install_revision_handler(
    layer: SQLiteChainlitDataLayer, handle_edit: Callable[[dict], Awaitable[None]]
) -> None:
    """Keep the native editor while ordering persistence in the application."""
    handlers = sio.handlers.get("/")
    original = handlers.get("edit_message") if handlers else None
    if original is None or tuple(inspect.signature(original).parameters) != (
        "sid",
        "payload",
    ):
        raise RuntimeError("Unsupported Chainlit edit_message signature")

    async def edit_message(sid, payload):
        session = WebsocketSession.require(sid)
        # An edit must not replace the task which the Stop button cancels.
        if session.current_task and not session.current_task.done():
            ctx = init_ws_context(session)
            await sync_chat_history(layer)
            await ctx.emitter.send_toast(
                "Дождитесь завершения текущего ответа.", "info"
            )
            return

        async def process() -> None:
            ctx = init_ws_context(session)
            await ctx.emitter.task_start()
            try:
                await handle_edit(payload)
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Could not revise Chat %s", session.thread_id)
                await ctx.emitter.send_toast("Не удалось изменить сообщение.", "error")
            finally:
                await ctx.emitter.task_end()

        session.current_task = asyncio.create_task(process())

    handlers["edit_message"] = edit_message
