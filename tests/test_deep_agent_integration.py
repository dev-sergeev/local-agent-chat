from pathlib import Path

import pytest
from deepagents.backends import StateBackend
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage

import local_agent_chat.agent_service as agent_module
from local_agent_chat.agent_events import TextDelta, ToolFinished, ToolStarted
from local_agent_chat.agent_modes import AgentMode
from local_agent_chat.agent_service import AgentService
from local_agent_chat.runtime import Turn
from local_agent_chat.sandbox_files import SandboxFiles
from local_agent_chat.sandbox_provider import LocalSandboxManager
from local_agent_chat.settings import ModelProfile
from local_agent_chat.sqlite_history import SQLiteHistory


class ToolAwareFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class StreamingToolAwareFakeModel(FakeListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class StateSandboxManager:
    def __init__(self) -> None:
        self.value = StateBackend()

    async def backend(self, chat_id: str, mode: AgentMode):
        return self.value

    def files_dir(self, chat_id: str) -> Path:
        return Path("/")

    async def push(self, chat_id: str, backend) -> None:
        return None

    async def pull(self, chat_id: str, backend) -> None:
        return None


@pytest.mark.asyncio
async def test_deep_agent_uses_sqlite_checkpoint_and_can_restart_from_empty_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = ToolAwareFakeModel(
        responses=[
            AIMessage(content="first"),
            AIMessage(content="second"),
            AIMessage(content="revised"),
        ]
    )
    monkeypatch.setattr(
        agent_module, "init_chat_model", lambda *args, **kwargs: fake_model
    )
    profile = ModelProfile("test", "Test", "openai:test", "TEST_KEY", "key")
    service = AgentService(
        tmp_path / "checkpoints.sqlite3", (profile,), StateSandboxManager()
    )  # type: ignore[arg-type]
    service.set_profile("chat-1", "test")

    before_first = await service.checkpoint("chat-1")
    assert await service.run("chat-1", "one") == "first"
    assert await service.run("chat-1", "two") == "second"
    await service.restore("chat-1", before_first)
    assert await service.run("chat-1", "changed") == "revised"
    await service.close()


@pytest.mark.asyncio
async def test_deep_agent_emits_safe_tool_lifecycle_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = ToolAwareFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/hello.txt", "content": "hello"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="finished"),
        ]
    )
    monkeypatch.setattr(
        agent_module, "init_chat_model", lambda *args, **kwargs: fake_model
    )
    profile = ModelProfile("test", "Test", "openai:test", "TEST_KEY", "key")
    service = AgentService(
        tmp_path / "checkpoints.sqlite3", (profile,), StateSandboxManager()
    )  # type: ignore[arg-type]
    service.set_profile("chat-1", "test")
    service.select_mode("chat-1", AgentMode.EXTENDED)
    events = []

    async def record(event) -> None:
        events.append(event)

    answer = await service.run("chat-1", "write a file", record)

    assert answer == "finished"
    assert isinstance(events[0], ToolStarted)
    assert events[0].name == "write_file"
    assert "/hello.txt" in events[0].input
    assert isinstance(events[1], ToolFinished)
    assert events[1].id == events[0].id
    assert "hello.txt" in events[1].output
    assert not any(isinstance(event, TextDelta) for event in events)
    await service.close()


@pytest.mark.asyncio
async def test_deep_agent_streams_public_answer_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        agent_module,
        "init_chat_model",
        lambda *args, **kwargs: StreamingToolAwareFakeModel(responses=["streamed"]),
    )
    profile = ModelProfile("test", "Test", "openai:test", "TEST_KEY", "key")
    service = AgentService(
        tmp_path / "checkpoints.sqlite3", (profile,), StateSandboxManager()
    )  # type: ignore[arg-type]
    service.set_profile("chat-1", "test")
    deltas: list[str] = []

    async def record(event) -> None:
        if isinstance(event, TextDelta):
            deltas.append(event.text)

    answer = await service.run("chat-1", "answer", record)

    assert answer == "streamed"
    assert "".join(deltas) == "streamed"
    await service.close()


@pytest.mark.asyncio
async def test_deep_agent_returns_complete_answer_when_streaming_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = StreamingToolAwareFakeModel(responses=["non-streamed answer"])
    model_kwargs = []

    def fake_init_chat_model(*args, **kwargs):
        model_kwargs.append(kwargs)
        fake_model.disable_streaming = kwargs.get("disable_streaming", False)
        return fake_model

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)
    profile = ModelProfile(
        "test",
        "Test",
        "openai:test",
        "TEST_KEY",
        "key",
        streaming=False,
    )
    service = AgentService(
        tmp_path / "checkpoints.sqlite3", (profile,), StateSandboxManager()
    )  # type: ignore[arg-type]
    service.set_profile("chat-1", "test")
    events = []

    async def record(event) -> None:
        events.append(event)

    answer = await service.run("chat-1", "answer", record)

    assert answer == "non-streamed answer"
    assert not any(isinstance(event, TextDelta) for event in events)
    assert model_kwargs[0]["disable_streaming"] is True
    await service.close()


@pytest.mark.asyncio
async def test_read_only_agent_reads_a_global_absolute_path_without_a_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = tmp_path / "outside-chat.txt"
    external.write_text("global host content", encoding="utf-8")
    fake_model = ToolAwareFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": str(external)},
                        "id": "read-global",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="read complete"),
        ]
    )
    monkeypatch.setattr(
        agent_module, "init_chat_model", lambda *_args, **_kwargs: fake_model
    )
    files = SandboxFiles(
        tmp_path / "sandboxes", max_file_bytes=1024, max_chat_bytes=4096
    )
    service = AgentService(
        tmp_path / "checkpoints.sqlite3",
        (ModelProfile("test", "Test", "openai:test", "TEST_KEY", "key"),),
        LocalSandboxManager(files),
    )
    service.set_profile("chat-1", "test")
    events = []

    async def record(event) -> None:
        events.append(event)

    answer = await service.run("chat-1", "read the absolute file", record)

    assert answer == "read complete"
    assert any(
        isinstance(event, ToolFinished) and "global host content" in event.output
        for event in events
    )
    assert not (tmp_path / "sandboxes" / "chat-1" / "environment").exists()
    await service.close()


@pytest.mark.asyncio
async def test_read_only_agent_cannot_dispatch_a_forced_write_tool_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "must-not-exist.txt"
    fake_model = ToolAwareFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": str(target), "content": "unsafe"},
                        "id": "forced-write",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="write unavailable"),
        ]
    )
    monkeypatch.setattr(
        agent_module, "init_chat_model", lambda *_args, **_kwargs: fake_model
    )
    files = SandboxFiles(
        tmp_path / "sandboxes", max_file_bytes=1024, max_chat_bytes=4096
    )
    service = AgentService(
        tmp_path / "checkpoints.sqlite3",
        (ModelProfile("test", "Test", "openai:test", "TEST_KEY", "key"),),
        LocalSandboxManager(files),
    )
    service.set_profile("chat-1", "test")

    answer = await service.run("chat-1", "try to write")

    assert answer == "write unavailable"
    assert target.exists() is False
    await service.close()


@pytest.mark.asyncio
async def test_read_only_agent_can_search_and_read_global_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = SQLiteHistory(tmp_path / "runtime-history.sqlite3")
    await history.append(
        Turn(
            "past-turn",
            "past-chat",
            "Решение по индексу памяти",
            "Использовать локальный SQLite FTS5",
            "memory",
            "files",
        )
    )
    fake_model = ToolAwareFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_past_chats",
                        "args": {"query": "индекс памяти"},
                        "id": "search-memory",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_past_chat",
                        "args": {
                            "chat_id": "past-chat",
                            "turn_id": "past-turn",
                            "context_turns": 1,
                        },
                        "id": "read-memory",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="В прошлый раз выбрали SQLite FTS5."),
        ]
    )
    monkeypatch.setattr(
        agent_module, "init_chat_model", lambda *_args, **_kwargs: fake_model
    )
    service = AgentService(
        tmp_path / "checkpoints.sqlite3",
        (ModelProfile("test", "Test", "openai:test", "TEST_KEY", "key"),),
        StateSandboxManager(),  # type: ignore[arg-type]
        global_memory=history,
    )
    service.set_profile("current-chat", "test")
    events = []

    async def record(event) -> None:
        events.append(event)

    answer = await service.run("current-chat", "Что мы выбрали раньше?", record)

    assert answer == "В прошлый раз выбрали SQLite FTS5."
    tool_outputs = [event.output for event in events if isinstance(event, ToolFinished)]
    assert any('"turn_id":"past-turn"' in output for output in tool_outputs)
    assert any("SQLite FTS5" in output for output in tool_outputs)
    await service.close()
