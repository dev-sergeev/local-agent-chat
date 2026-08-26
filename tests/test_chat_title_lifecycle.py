from __future__ import annotations

import asyncio
import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any

import chainlit as cl
import chainlit.data as chainlit_data_runtime
import pytest
from chainlit.context import ChainlitContext, context_var
from chainlit.emitter import BaseChainlitEmitter
from chainlit.server import sio
from chainlit.session import HTTPSession, WebsocketSession
from chainlit.user import User
from langchain import chat_models as langchain_chat_models
from langchain_core.messages import AIMessage

from local_agent_chat.agent_events import EventSink
from local_agent_chat.chat_titles import (
    CHAT_TITLE_GENERATED,
    CHAT_TITLE_STATE_KEY,
)
from local_agent_chat.runtime import ChatRuntime
from local_agent_chat.sandbox_files import SandboxFiles
from local_agent_chat.sqlite_history import SQLiteHistory

EXPECTED_TITLE = "Запуск скрипта и проверка файлов"


class TransientTitleModel:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary title provider failure")
        return AIMessage(content=EXPECTED_TITLE)


class AnsweringAgent:
    async def checkpoint(self, _chat_id: str) -> str:
        return "memory-before-turn"

    async def restore(self, _chat_id: str, _checkpoint: str) -> None:
        return None

    async def run(
        self, _chat_id: str, _text: str, _emit: EventSink | None = None
    ) -> str:
        return "Готово"


class RecordingEmitter(BaseChainlitEmitter):
    def __init__(self, session: HTTPSession) -> None:
        super().__init__(session)
        self.title_events: list[str] = []
        self.chat_settings_events: list[list[dict[str, Any]]] = []

    def set_chat_settings(self, settings: dict) -> None:
        self.session.chat_settings = settings

    async def emit(self, event: str, data: Any) -> None:
        if event == "first_interaction":
            self.title_events.append(str(data["interaction"]))
        elif event == "chat_settings":
            self.chat_settings_events.append(data)


async def _wait_for_title_attempt(layer, expected_state: str) -> None:
    for _ in range(100):
        if await layer.chat_title_state("chat-1") == expected_state:
            return
        await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_transient_chat_title_failure_retries_on_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "app-data"))
    monkeypatch.setenv(
        "MODEL_PROFILES_FILE", str(Path("models.example.yaml").resolve())
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    title_model = TransientTitleModel()
    monkeypatch.setattr(
        langchain_chat_models,
        "init_chat_model",
        lambda *_args, **_kwargs: title_model,
    )
    module_name = "_chat_title_lifecycle_app"
    app_path = Path(__file__).parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    assert spec is not None and spec.loader is not None
    chat_app = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = chat_app
    spec.loader.exec_module(chat_app)

    layer = chat_app.chainlit_layer
    monkeypatch.setattr(chainlit_data_runtime, "_data_layer", layer)
    monkeypatch.setattr(chainlit_data_runtime, "_data_layer_initialized", True)
    history = SQLiteHistory(tmp_path / "runtime.sqlite3")
    sandbox = SandboxFiles(
        tmp_path / "sandboxes",
        max_file_bytes=1024,
        max_chat_bytes=4096,
    )
    chat_app.runtime_history = history
    chat_app.sandbox_files = sandbox
    chat_app.runtime = ChatRuntime(
        agent=AnsweringAgent(),
        sandbox=sandbox,
        history=history,
    )

    user = await layer.create_user(User(identifier="local-user", metadata={}))
    assert user is not None
    session = HTTPSession(
        id="session-1",
        client_type="webapp",
        thread_id="chat-1",
        user=user,
    )
    emitter = RecordingEmitter(session)
    context_var.set(ChainlitContext(session, emitter))
    first_request = "Создай Python-скрипт, запусти его и проверь список файлов"

    try:
        await chat_app.on_chat_start()
        initial_mode = next(
            item
            for item in emitter.chat_settings_events[-1]
            if item["id"] == "extended_mode"
        )
        assert initial_mode["initial"] is False
        assert initial_mode["disabled"] is False

        await chat_app.on_settings_update(
            {"extended_mode": True, "show_tool_details": False}
        )
        binding = chat_app.chat_bindings.get("chat-1")
        assert binding is not None
        assert binding.mode.value == "extended"
        assert binding.mode_locked is False

        await layer.update_thread("chat-1", name=first_request, user_id=user.id)
        first = cl.Message(id="turn-1", content=first_request, type="user_message")
        await layer.create_step(first.to_dict())
        await chat_app.on_message(first)
        await _wait_for_title_attempt(layer, "fallback")
        locked_mode = next(
            item
            for item in emitter.chat_settings_events[-1]
            if item["id"] == "extended_mode"
        )
        assert locked_mode["initial"] is True
        assert locked_mode["disabled"] is True
        await chat_app.on_settings_update(
            {"extended_mode": False, "show_tool_details": False}
        )
        binding = chat_app.chat_bindings.get("chat-1")
        assert binding is not None
        assert binding.mode.value == "extended"

        second = cl.Message(id="turn-2", content="Продолжай", type="user_message")
        await layer.create_step(second.to_dict())
        await chat_app.on_message(second)
        await _wait_for_title_attempt(layer, CHAT_TITLE_GENERATED)

        thread = await layer.get_thread("chat-1")
        visible_title = emitter.title_events[-1]
        assert thread is not None
        assert (
            thread["name"],
            thread["metadata"][CHAT_TITLE_STATE_KEY],
            visible_title,
            title_model.calls,
        ) == (
            EXPECTED_TITLE,
            CHAT_TITLE_GENERATED,
            EXPECTED_TITLE,
            2,
        )
    finally:
        pending = list(chat_app.chat_title_tasks.values())
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await chat_app.agent_execution.close()
        await layer.close()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery_entrypoint", ["resume", "settings", "stop"])
async def test_persisted_first_message_locks_mode_at_next_ui_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery_entrypoint: str,
) -> None:
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "app-data"))
    monkeypatch.setenv(
        "MODEL_PROFILES_FILE", str(Path("models.example.yaml").resolve())
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    module_name = f"_agent_mode_{recovery_entrypoint}_app"
    app_path = Path(__file__).parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    assert spec is not None and spec.loader is not None
    chat_app = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = chat_app
    spec.loader.exec_module(chat_app)

    layer = chat_app.chainlit_layer
    monkeypatch.setattr(chainlit_data_runtime, "_data_layer", layer)
    monkeypatch.setattr(chainlit_data_runtime, "_data_layer_initialized", True)
    user = await layer.create_user(User(identifier="local-user", metadata={}))
    assert user is not None
    session = HTTPSession(
        id="session-resumed",
        client_type="webapp",
        thread_id="chat-resumed",
        user=user,
    )
    emitter = RecordingEmitter(session)
    context_var.set(ChainlitContext(session, emitter))
    request = "Первое сообщение уже сохранено"

    try:
        await chat_app.on_chat_start()
        await chat_app.on_settings_update(
            {"extended_mode": True, "show_tool_details": False}
        )
        await layer.update_thread("chat-resumed", name=request, user_id=user.id)
        assert await layer.complete_chat_title(
            "chat-resumed", "Сохранённое первое сообщение"
        )
        first = cl.Message(id="turn-resumed", content=request, type="user_message")
        await layer.create_step(first.to_dict())
        binding = chat_app.chat_bindings.get("chat-resumed")
        assert binding is not None
        assert binding.mode_locked is False

        if recovery_entrypoint == "resume":
            await chat_app.on_chat_resume(
                {"id": "chat-resumed", "metadata": {"model_profile": "local"}}
            )
        elif recovery_entrypoint == "settings":
            await chat_app.on_settings_update(
                {"extended_mode": False, "show_tool_details": False}
            )
        else:
            await chat_app.on_stop()

        resumed_mode = next(
            item
            for item in emitter.chat_settings_events[-1]
            if item["id"] == "extended_mode"
        )
        assert resumed_mode["initial"] is True
        assert resumed_mode["disabled"] is True
        binding = chat_app.chat_bindings.get("chat-resumed")
        assert binding is not None
        assert binding.mode.value == "extended"
        assert binding.mode_locked is True
        resumed_thread = await layer.get_thread("chat-resumed")
        assert resumed_thread is not None
        assert resumed_thread["metadata"]["agent_mode"] == "extended"
        assert resumed_thread["metadata"]["agent_mode_locked"] is True
        assert resumed_thread["metadata"][CHAT_TITLE_STATE_KEY] == CHAT_TITLE_GENERATED
    finally:
        await chat_app.agent_execution.close()
        await layer.close()
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_order", "expected_mode"),
    [
        ("message_then_read_only", "extended"),
        ("read_only_then_message", "read_only"),
        ("invalid_then_read_only_then_message", "read_only"),
    ],
)
async def test_socket_acceptance_locks_mode_before_background_message_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    event_order: str,
    expected_mode: str,
) -> None:
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "app-data"))
    monkeypatch.setenv(
        "MODEL_PROFILES_FILE", str(Path("models.example.yaml").resolve())
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    module_name = "_agent_mode_socket_acceptance_app"
    app_path = Path(__file__).parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    assert spec is not None and spec.loader is not None
    chat_app = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = chat_app
    spec.loader.exec_module(chat_app)

    chat_id = "chat-socket-race"
    socket_id = "socket-race"
    session = WebsocketSession(
        id="session-socket-race",
        socket_id=socket_id,
        emit=lambda *_args, **_kwargs: None,
        emit_call=lambda *_args, **_kwargs: None,
        user_env={},
        client_type="webapp",
        thread_id=chat_id,
    )
    processing_started = asyncio.Event()
    release_processing = asyncio.Event()
    background_tasks: list[asyncio.Task[None]] = []

    async def blocked_event_handler(*_args) -> None:
        processing_started.set()
        await release_processing.wait()

    def start_background_task(target, *args):
        task = asyncio.create_task(target(*args))
        background_tasks.append(task)
        return task

    monkeypatch.setattr(sio.manager, "sid_from_eio_sid", lambda *_args: socket_id)
    monkeypatch.setattr(sio.manager, "is_connected", lambda *_args: True)
    monkeypatch.setattr(sio, "_handle_event_internal", blocked_event_handler)
    monkeypatch.setattr(sio, "start_background_task", start_background_task)

    try:
        await sio._handle_event(
            "engine-socket-race",
            "/",
            None,
            ["chat_settings_change", {"extended_mode": True}],
        )
        read_only_event = ["chat_settings_change", {"extended_mode": False}]
        message_event = [
            "client_message",
            {
                "message": {
                    "id": str(uuid.uuid4()),
                    "createdAt": "2026-08-25T17:00:00Z",
                    "output": "Первое сообщение",
                }
            },
        ]
        if event_order == "message_then_read_only":
            ordered_events = (message_event, read_only_event)
        elif event_order == "read_only_then_message":
            ordered_events = (read_only_event, message_event)
        else:
            invalid_message = [
                "client_message",
                {
                    "message": {
                        "id": 12345678123442348234123456789012,
                        "createdAt": "2026-08-25T17:00:00Z",
                        "output": "Некорректное сообщение",
                    }
                },
            ]
            ordered_events = (invalid_message, read_only_event, message_event)
        for event in ordered_events:
            await sio._handle_event("engine-socket-race", "/", None, event)
        await asyncio.sleep(0)
        await asyncio.wait_for(processing_started.wait(), timeout=1)

        binding = chat_app.chat_bindings.get(chat_id)
        assert binding is not None
        assert binding.mode.value == expected_mode
        assert binding.mode_locked is True
        assert await chat_app.chainlit_layer.has_user_request(chat_id) is False
        assert len(background_tasks) == len(ordered_events) + 1
        assert all(not task.done() for task in background_tasks)
    finally:
        release_processing.set()
        await asyncio.gather(*background_tasks, return_exceptions=True)
        await session.delete()
        await chat_app.agent_execution.close()
        await chat_app.chainlit_layer.close()
        sys.modules.pop(module_name, None)
