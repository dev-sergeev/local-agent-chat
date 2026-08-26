from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
from chainlit.context import ChainlitContext, context_var
from chainlit.emitter import BaseChainlitEmitter
from chainlit.server import sio
from chainlit.session import HTTPSession, WebsocketSession


class SilentEmitter(BaseChainlitEmitter):
    def set_chat_settings(self, settings: dict) -> None:
        self.session.chat_settings = settings

    async def emit(self, _event: str, _data: Any) -> None:
        return None


def _profiles_file(tmp_path: Path) -> Path:
    profiles = tmp_path / "models.yaml"
    profiles.write_text(
        """models:
  - id: first
    label: First
    model: openai:first
    api_key_env: TEST_FIRST_KEY
  - id: second
    label: Second
    model: openai:second
    api_key_env: TEST_SECOND_KEY
""",
        encoding="utf-8",
    )
    return profiles


def _load_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str):
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "app-data"))
    monkeypatch.setenv("MODEL_PROFILES_FILE", str(_profiles_file(tmp_path)))
    app_path = Path(__file__).parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    assert spec is not None and spec.loader is not None
    chat_app = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = chat_app
    spec.loader.exec_module(chat_app)
    return chat_app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_profile", "expected_profile"),
    [("local", "first"), ("second", "second")],
)
async def test_chat_start_falls_back_from_a_removed_client_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client_profile: str,
    expected_profile: str,
) -> None:
    module_name = f"_model_profile_start_{client_profile}_app"
    chat_app = _load_app(tmp_path, monkeypatch, module_name)
    session = HTTPSession(
        id="profile-start-session",
        client_type="webapp",
        thread_id="chat-start",
    )
    session.chat_profile = client_profile
    context_var.set(ChainlitContext(session, SilentEmitter(session)))

    try:
        await chat_app.on_chat_start()

        binding = chat_app.chat_bindings.get("chat-start")
        assert binding is not None
        assert binding.profile_id == expected_profile
        assert chat_app.cl.user_session.get("model_profile") == expected_profile
    finally:
        await chat_app.agent_execution.close()
        await chat_app.chainlit_layer.close()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_socket_acceptance_falls_back_from_a_removed_client_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "_model_profile_socket_app"
    chat_app = _load_app(tmp_path, monkeypatch, module_name)
    chat_id = "chat-socket"
    socket_id = "socket-profile"
    session = WebsocketSession(
        id="profile-socket-session",
        socket_id=socket_id,
        emit=lambda *_args, **_kwargs: None,
        emit_call=lambda *_args, **_kwargs: None,
        user_env={},
        client_type="webapp",
        thread_id=chat_id,
    )
    session.chat_profile = "local"

    async def ignored_event_handler(*_args) -> None:
        return None

    monkeypatch.setattr(sio.manager, "sid_from_eio_sid", lambda *_args: socket_id)
    monkeypatch.setattr(sio.manager, "is_connected", lambda *_args: True)
    monkeypatch.setattr(sio, "async_handlers", False)
    monkeypatch.setattr(sio, "_handle_event_internal", ignored_event_handler)

    try:
        await sio._handle_event(
            "engine-profile",
            "/",
            None,
            [
                "client_message",
                {
                    "message": {
                        "id": str(uuid.uuid4()),
                        "createdAt": "2026-08-25T17:00:00Z",
                        "output": "Первое сообщение",
                    }
                },
            ],
        )

        binding = chat_app.chat_bindings.get(chat_id)
        assert binding is not None
        assert binding.profile_id == "first"
        assert binding.mode_locked is True
    finally:
        await session.delete()
        await chat_app.agent_execution.close()
        await chat_app.chainlit_layer.close()
        sys.modules.pop(module_name, None)
