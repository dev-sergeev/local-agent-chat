from pathlib import Path

import pytest
from deepagents.backends import StateBackend
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_openai import StreamChunkTimeoutError

import local_agent_chat.deep_agent_execution as execution_module
from local_agent_chat.agent_events import TextDelta, ToolFinished, ToolStarted
from local_agent_chat.agent_modes import AgentMode
from local_agent_chat.chat_bindings import ChatBindings
from local_agent_chat.deep_agent_execution import DeepAgentExecution
from local_agent_chat.runtime import Turn
from local_agent_chat.sandbox_files import SandboxFiles
from local_agent_chat.sandbox_provider import LocalSandboxManager
from local_agent_chat.settings import LLMRetryConfig, ModelProfile
from local_agent_chat.sqlite_history import SQLiteHistory


class ToolAwareFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class StreamingToolAwareFakeModel(FakeListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class RecoveringSubagentFakeModel(ToolAwareFakeModel):
    calls: int = 0

    def _generate(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 2:
            raise StreamChunkTimeoutError(0.01, chunks_received=0)
        return super()._generate(*args, **kwargs)


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
async def test_retry_configured_model_reaches_default_subagent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = RecoveringSubagentFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "Return a short reliability report",
                            "subagent_type": "general-purpose",
                        },
                        "id": "delegate-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="subagent report"),
            AIMessage(content="main agent result"),
        ]
    )
    init_kwargs: list[dict[str, object]] = []

    def fake_init_chat_model(*_args, **kwargs):
        init_kwargs.append(kwargs)
        return fake_model

    monkeypatch.setattr(execution_module, "init_chat_model", fake_init_chat_model)
    profile = ModelProfile("test", "Test", "openai:test", "TEST_KEY", "key")
    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, (profile.id,))
    bindings.open("chat-1", profile.id)
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile,),
        StateSandboxManager(),  # type: ignore[arg-type]
        llm_retry=LLMRetryConfig(max_retries=6, stream_retries=1),
        chat_bindings=bindings,
    )
    events = []

    async def record(event) -> None:
        events.append(event)

    assert await execution.run("chat-1", "delegate", record) == "main agent result"
    assert fake_model.calls == 4
    assert (
        sum(isinstance(event, ToolStarted) and event.name == "task" for event in events)
        == 1
    )
    assert init_kwargs == [
        {
            "api_key": "key",
            "max_retries": 6,
            "timeout": 60.0,
            "stream_chunk_timeout": 120.0,
        }
    ]
    await execution.close()


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
        execution_module, "init_chat_model", lambda *args, **kwargs: fake_model
    )
    profile = ModelProfile("test", "Test", "openai:test", "TEST_KEY", "key")
    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, (profile.id,))
    bindings.open("chat-1", profile.id)
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile,),
        StateSandboxManager(),
        chat_bindings=bindings,
    )  # type: ignore[arg-type]

    before_first = await execution.checkpoint("chat-1")
    assert await execution.run("chat-1", "one") == "first"
    assert await execution.run("chat-1", "two") == "second"
    await execution.restore("chat-1", before_first)
    assert await execution.run("chat-1", "changed") == "revised"
    await execution.close()


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
        execution_module, "init_chat_model", lambda *args, **kwargs: fake_model
    )
    profile = ModelProfile("test", "Test", "openai:test", "TEST_KEY", "key")
    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, (profile.id,))
    bindings.open("chat-1", profile.id)
    bindings.select_mode("chat-1", AgentMode.EXTENDED)
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile,),
        StateSandboxManager(),
        chat_bindings=bindings,
    )  # type: ignore[arg-type]
    events = []

    async def record(event) -> None:
        events.append(event)

    answer = await execution.run("chat-1", "write a file", record)

    assert answer == "finished"
    assert isinstance(events[0], ToolStarted)
    assert events[0].name == "write_file"
    assert "/hello.txt" in events[0].input
    assert isinstance(events[1], ToolFinished)
    assert events[1].id == events[0].id
    assert "hello.txt" in events[1].output
    assert not any(isinstance(event, TextDelta) for event in events)
    await execution.close()


@pytest.mark.asyncio
async def test_deep_agent_streams_public_answer_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        execution_module,
        "init_chat_model",
        lambda *args, **kwargs: StreamingToolAwareFakeModel(responses=["streamed"]),
    )
    profile = ModelProfile("test", "Test", "openai:test", "TEST_KEY", "key")
    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, (profile.id,))
    bindings.open("chat-1", profile.id)
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile,),
        StateSandboxManager(),
        chat_bindings=bindings,
    )  # type: ignore[arg-type]
    deltas: list[str] = []

    async def record(event) -> None:
        if isinstance(event, TextDelta):
            deltas.append(event.text)

    answer = await execution.run("chat-1", "answer", record)

    assert answer == "streamed"
    assert "".join(deltas) == "streamed"
    await execution.close()


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

    monkeypatch.setattr(execution_module, "init_chat_model", fake_init_chat_model)
    profile = ModelProfile(
        "test",
        "Test",
        "openai:test",
        "TEST_KEY",
        "key",
        streaming=False,
    )
    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, (profile.id,))
    bindings.open("chat-1", profile.id)
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile,),
        StateSandboxManager(),
        chat_bindings=bindings,
    )  # type: ignore[arg-type]
    events = []

    async def record(event) -> None:
        events.append(event)

    answer = await execution.run("chat-1", "answer", record)

    assert answer == "non-streamed answer"
    assert not any(isinstance(event, TextDelta) for event in events)
    assert model_kwargs[0]["disable_streaming"] is True
    await execution.close()


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
        execution_module, "init_chat_model", lambda *_args, **_kwargs: fake_model
    )
    files = SandboxFiles(
        tmp_path / "sandboxes", max_file_bytes=1024, max_chat_bytes=4096
    )
    profile = ModelProfile("test", "Test", "openai:test", "TEST_KEY", "key")
    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, (profile.id,))
    bindings.open("chat-1", profile.id)
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile,),
        LocalSandboxManager(files),
        chat_bindings=bindings,
    )
    events = []

    async def record(event) -> None:
        events.append(event)

    answer = await execution.run("chat-1", "read the absolute file", record)

    assert answer == "read complete"
    assert any(
        isinstance(event, ToolFinished) and "global host content" in event.output
        for event in events
    )
    assert not (tmp_path / "sandboxes" / "chat-1" / "environment").exists()
    await execution.close()


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
        execution_module, "init_chat_model", lambda *_args, **_kwargs: fake_model
    )
    files = SandboxFiles(
        tmp_path / "sandboxes", max_file_bytes=1024, max_chat_bytes=4096
    )
    profile = ModelProfile("test", "Test", "openai:test", "TEST_KEY", "key")
    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, (profile.id,))
    bindings.open("chat-1", profile.id)
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile,),
        LocalSandboxManager(files),
        chat_bindings=bindings,
    )

    answer = await execution.run("chat-1", "try to write")

    assert answer == "write unavailable"
    assert target.exists() is False
    await execution.close()


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
        execution_module, "init_chat_model", lambda *_args, **_kwargs: fake_model
    )
    profile = ModelProfile("test", "Test", "openai:test", "TEST_KEY", "key")
    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, (profile.id,))
    bindings.open("current-chat", profile.id)
    bindings.lock("current-chat")
    execution = DeepAgentExecution(
        database,
        (profile,),
        StateSandboxManager(),  # type: ignore[arg-type]
        global_memory=history,
        chat_bindings=bindings,
    )
    events = []

    async def record(event) -> None:
        events.append(event)

    answer = await execution.run("current-chat", "Что мы выбрали раньше?", record)

    assert answer == "В прошлый раз выбрали SQLite FTS5."
    tool_outputs = [event.output for event in events if isinstance(event, ToolFinished)]
    assert any('"turn_id":"past-turn"' in output for output in tool_outputs)
    assert any("SQLite FTS5" in output for output in tool_outputs)
    await execution.close()
