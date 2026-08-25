from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from local_agent_chat import agent_service as agent_service_module
from local_agent_chat.agent_service import AgentService
from local_agent_chat.settings import ModelProfile


class UnusedSandboxes:
    async def backend(self, chat_id: str):
        raise AssertionError("backend should not be created while selecting a profile")


def profile() -> ModelProfile:
    return ModelProfile("local", "Local", "openai:test", "TEST_KEY", "key")


def test_model_profile_is_immutable_and_survives_restart(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    service = AgentService(database, (profile(),), UnusedSandboxes())  # type: ignore[arg-type]
    service.set_profile("chat-1", "local")

    reopened = AgentService(database, (profile(),), UnusedSandboxes())  # type: ignore[arg-type]
    reopened.set_profile("chat-1", "local")
    with pytest.raises(ValueError, match="Unknown Model Profile"):
        reopened.set_profile("chat-1", "missing")


@pytest.mark.asyncio
async def test_deleting_chat_removes_persisted_model_profile(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    service = AgentService(database, (profile(),), UnusedSandboxes())  # type: ignore[arg-type]
    service.set_profile("chat-1", "local")
    await service.delete_chat("chat-1")
    await service.close()

    reopened = AgentService(database, (profile(),), UnusedSandboxes())  # type: ignore[arg-type]
    assert reopened.profile_for("chat-1") is None


@pytest.mark.asyncio
async def test_describe_tool_uses_chat_profile_and_normalizes_title(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    class FakeModel:
        async def ainvoke(self, messages):
            calls.append(messages)
            return AIMessage(content="Изучаю параметры текущей системы")

    init_calls = []

    def fake_init(model, **kwargs):
        init_calls.append((model, kwargs))
        return FakeModel()

    monkeypatch.setattr(agent_service_module, "init_chat_model", fake_init)
    service = AgentService(
        tmp_path / "checkpoints.sqlite3",
        (profile(),),
        UnusedSandboxes(),  # type: ignore[arg-type]
    )
    service.set_profile("chat-1", "local")

    title = await service.describe_tool(
        "chat-1", "execute", '{"command":"whoami; uname -a"}'
    )

    assert title == "Изучаю параметры текущей системы"
    assert init_calls == [
        (
            "openai:test",
            {
                "api_key": "key",
                "max_tokens": 32,
                "reasoning_effort": "none",
            },
        )
    ]
    assert "whoami; uname -a" in calls[0][1][1]


@pytest.mark.asyncio
async def test_describe_tool_failure_does_not_escape(
    tmp_path: Path, monkeypatch
) -> None:
    class FailingModel:
        async def ainvoke(self, _messages):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        agent_service_module,
        "init_chat_model",
        lambda *_args, **_kwargs: FailingModel(),
    )
    service = AgentService(
        tmp_path / "checkpoints.sqlite3",
        (profile(),),
        UnusedSandboxes(),  # type: ignore[arg-type]
    )
    service.set_profile("chat-1", "local")

    assert await service.describe_tool("chat-1", "execute", "pwd") is None
