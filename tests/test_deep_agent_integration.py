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
from local_agent_chat.agent_service import AgentService
from local_agent_chat.settings import ModelProfile


class ToolAwareFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class StreamingToolAwareFakeModel(FakeListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class StateSandboxManager:
    def __init__(self) -> None:
        self.value = StateBackend()

    async def backend(self, chat_id: str):
        return self.value

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
