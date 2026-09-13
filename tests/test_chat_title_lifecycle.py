from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import chainlit as cl
import chainlit.data as chainlit_data_runtime
import pytest
from chainlit.context import ChainlitContext, context_var
from chainlit.emitter import BaseChainlitEmitter
from chainlit.session import HTTPSession
from chainlit.user import User
from langchain import chat_models as langchain_chat_models
from langchain_core.messages import AIMessage

from local_agent_chat.agent_events import EventSink
from local_agent_chat.chat_titles import (
    CHAT_TITLE_GENERATED,
    CHAT_TITLE_STATE_KEY,
    fallback_chat_title,
)
from local_agent_chat.runtime import ChatRuntime
from local_agent_chat.sandbox_files import SandboxFiles
from local_agent_chat.sqlite_history import SQLiteHistory

EXPECTED_TITLE = "Анализ загруженных файлов"


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
    first_request = "Изучи загруженные файлы и объясни их назначение"

    try:
        await chat_app.on_chat_start()
        assert {item["id"] for item in emitter.chat_settings_events[-1]} == {
            "show_tool_details"
        }
        # Stale clients cannot re-enable the removed host-files capability.
        await chat_app.on_settings_update(
            {"host_files_access": True, "show_tool_details": False}
        )

        await layer.update_thread("chat-1", name=first_request, user_id=user.id)
        first = cl.Message(id="turn-1", content=first_request, type="user_message")
        await layer.create_step(first.to_dict())
        await chat_app.on_message(first)
        await _wait_for_title_attempt(layer, "fallback")
        fallback_thread = await layer.get_thread("chat-1")
        assert fallback_thread is not None
        assert fallback_thread["name"] == fallback_chat_title(first_request)
        assert emitter.title_events[-1] == fallback_chat_title(first_request)
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
