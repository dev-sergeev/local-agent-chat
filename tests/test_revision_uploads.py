from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chainlit as cl
import chainlit.data as chainlit_data_runtime
import pytest
from chainlit.context import ChainlitContext, context_var
from chainlit.emitter import BaseChainlitEmitter
from chainlit.session import HTTPSession
from chainlit.user import User

from local_agent_chat.agent_events import EventSink
from local_agent_chat.runtime import ChatRuntime
from local_agent_chat.sandbox_files import SandboxFiles
from local_agent_chat.sqlite_history import SQLiteHistory


class FileReadingAgent:
    def __init__(self, files: SandboxFiles) -> None:
        self._files = files
        self.seen_files: list[list[str]] = []

    async def checkpoint(self, _chat_id: str) -> str:
        return "memory-before-turn"

    async def restore(self, _chat_id: str, _checkpoint: str) -> None:
        return None

    async def run(
        self, chat_id: str, _text: str, _emit: EventSink | None = None
    ) -> str:
        names = sorted(self._files.manifest(chat_id))
        self.seen_files.append(names)
        return ", ".join(names) or "no files"


class RecordingEmitter(BaseChainlitEmitter):
    def set_chat_settings(self, settings: dict) -> None:
        self.session.chat_settings = settings

    async def emit(self, _event: str, _data: Any) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("commit_failure", [False, True])
async def test_revised_message_upload_is_available_to_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commit_failure: bool,
) -> None:
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "app-data"))
    monkeypatch.setenv(
        "MODEL_PROFILES_FILE", str(Path("models.example.yaml").resolve())
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    module_name = f"_revision_upload_{commit_failure}_app"
    app_path = Path(__file__).parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    assert spec is not None and spec.loader is not None
    chat_app = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = chat_app
    spec.loader.exec_module(chat_app)

    layer = chat_app.chainlit_layer
    monkeypatch.setattr(chainlit_data_runtime, "_data_layer", layer)
    monkeypatch.setattr(chainlit_data_runtime, "_data_layer_initialized", True)
    monkeypatch.setattr(chat_app, "_start_chat_title", lambda *_args: None)
    files = SandboxFiles(
        tmp_path / "sandboxes",
        max_file_bytes=1024,
        max_chat_bytes=4096,
    )
    history = SQLiteHistory(tmp_path / "runtime.sqlite3")
    agent = FileReadingAgent(files)
    chat_app.sandbox_files = files
    chat_app.runtime_history = history
    chat_app.runtime = ChatRuntime(agent=agent, sandbox=files, history=history)

    user = await layer.create_user(User(identifier="local-user", metadata={}))
    assert user is not None
    session = HTTPSession(
        id="session-1",
        client_type="webapp",
        thread_id="chat-1",
        user=user,
    )
    context_var.set(ChainlitContext(session, RecordingEmitter(session)))

    try:
        await chat_app.on_chat_start()
        await layer.update_thread("chat-1", name="Original", user_id=user.id)
        original = cl.Message(
            id="turn-1",
            content="Original",
            type="user_message",
            created_at="2026-08-26T00:00:00Z",
        )
        await layer.create_step(original.to_dict())
        await chat_app.on_message(original)

        source = tmp_path / "revision.txt"
        source.write_text("attached to revision", encoding="utf-8")
        revised = cl.Message(
            id="turn-1",
            content="Read the revision file",
            type="user_message",
            created_at="2026-08-26T00:00:00Z",
        )
        await layer.update_step(revised.to_dict())
        revised.elements = [SimpleNamespace(path=str(source), name=source.name)]

        if commit_failure:

            async def fail_commit(_root_id: str) -> None:
                raise RuntimeError("UI commit failed")

            monkeypatch.setattr(layer, "_commit_revision", fail_commit)
            with pytest.raises(RuntimeError, match="UI commit failed"):
                await chat_app.on_message(revised)
        else:
            await chat_app.on_message(revised)

        assert agent.seen_files == [[], ["revision.txt"]]
        if commit_failure:
            assert sorted(files.manifest("chat-1")) == []
            assert (await history.get("turn-1")).text == "Original"
            restored_user_step = await layer.get_step("turn-1")
            assert restored_user_step is not None
            assert restored_user_step["output"] == "Original"
        else:
            assert sorted(files.manifest("chat-1")) == ["revision.txt"]
            assert (await history.get("turn-1")).text == "Read the revision file"
    finally:
        pending = list(chat_app.chat_title_tasks.values())
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await chat_app.agent_execution.close()
        await layer.close()
        sys.modules.pop(module_name, None)
