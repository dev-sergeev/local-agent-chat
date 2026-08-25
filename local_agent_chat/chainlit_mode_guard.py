from __future__ import annotations

import inspect
import uuid
from collections.abc import Callable
from typing import Any

from chainlit.server import sio
from chainlit.session import WebsocketSession

_ORIGINAL_HANDLER = "_local_agent_chat_original_handle_event"
_EXPECTED_PARAMETERS = ("eio_sid", "namespace", "id", "data")


def _valid_user_message(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    message = payload.get("message")
    if not isinstance(message, dict):
        return False
    if not {"id", "createdAt", "output"} <= message.keys():
        return False
    message_id = message["id"]
    if not isinstance(message_id, str):
        return False
    try:
        return uuid.UUID(message_id).version == 4
    except (ValueError, TypeError, AttributeError):
        return False


def install_mode_acceptance_guard(
    select_mode: Callable[[str, str | None, bool], None],
    lock_chat: Callable[[str, str | None], None],
) -> None:
    """Order mode selection and first-message locking before async dispatch."""

    # Chainlit's public Socket.IO handlers run in independent tasks. This private
    # pre-dispatch seam is where Engine.IO packet order is still authoritative.
    current_handler = sio._handle_event
    original_handler = getattr(current_handler, _ORIGINAL_HANDLER, current_handler)
    if tuple(inspect.signature(original_handler).parameters) != _EXPECTED_PARAMETERS:
        raise RuntimeError("Unsupported python-socketio _handle_event signature")

    async def guarded_handle_event(
        eio_sid: str,
        namespace: str | None,
        id: str | int | None,
        data: list[Any],
    ) -> None:
        active_namespace = namespace or "/"
        event = data[0] if isinstance(data, list) and data else None
        payload = data[1] if isinstance(data, list) and len(data) > 1 else None
        if event in {"chat_settings_change", "client_message"}:
            sid = sio.manager.sid_from_eio_sid(eio_sid, active_namespace)
            session = WebsocketSession.get(sid) if sid is not None else None
            if session is not None and sio.manager.is_connected(sid, active_namespace):
                if (
                    event == "chat_settings_change"
                    and isinstance(payload, dict)
                    and "extended_mode" in payload
                ):
                    select_mode(
                        session.thread_id,
                        session.chat_profile,
                        payload["extended_mode"] is True,
                    )
                elif event == "client_message" and _valid_user_message(payload):
                    lock_chat(session.thread_id, session.chat_profile)

        await original_handler(eio_sid, namespace, id, data)

    setattr(guarded_handle_event, _ORIGINAL_HANDLER, original_handler)
    sio._handle_event = guarded_handle_event
