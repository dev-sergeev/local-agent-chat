import asyncio
import sqlite3
from pathlib import Path

import pytest
from deepagents.backends import StateBackend
from langchain_core.messages import AIMessage

from local_agent_chat import agent_service as agent_service_module
from local_agent_chat.agent_modes import (
    EXTENDED_FILESYSTEM_TOOLS,
    READ_ONLY_FILESYSTEM_TOOLS,
    AgentMode,
)
from local_agent_chat.agent_service import AgentService
from local_agent_chat.settings import ModelProfile
from local_agent_chat.sqlite_history import SQLiteHistory


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


def test_agent_mode_defaults_to_read_only_and_locks_at_first_turn(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    service = AgentService(database, (profile(),), UnusedSandboxes())  # type: ignore[arg-type]
    service.set_profile("chat-1", "local")

    assert service.mode_for("chat-1") is AgentMode.READ_ONLY
    assert service.mode_is_locked("chat-1") is False
    assert service.select_mode("chat-1", AgentMode.EXTENDED) is AgentMode.EXTENDED
    assert service.lock_mode("chat-1") is AgentMode.EXTENDED
    assert service.mode_is_locked("chat-1") is True
    assert service.select_mode("chat-1", AgentMode.EXTENDED) is AgentMode.EXTENDED
    with pytest.raises(ValueError, match="cannot change"):
        service.select_mode("chat-1", AgentMode.READ_ONLY)

    reopened = AgentService(database, (profile(),), UnusedSandboxes())  # type: ignore[arg-type]
    assert reopened.mode_for("chat-1") is AgentMode.EXTENDED
    assert reopened.mode_is_locked("chat-1") is True


def test_legacy_chat_is_migrated_to_locked_extended_mode(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE active_branches (chat_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO active_branches VALUES ('legacy-chat', 'local', '')"
        )

    service = AgentService(database, (profile(),), UnusedSandboxes())  # type: ignore[arg-type]

    assert service.mode_for("legacy-chat") is AgentMode.EXTENDED
    assert service.mode_is_locked("legacy-chat") is True


def test_invalid_persisted_mode_fails_closed_to_read_only(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """CREATE TABLE active_branches (
                   chat_id TEXT PRIMARY KEY,
                   profile_id TEXT NOT NULL,
                   checkpoint_ns TEXT NOT NULL,
                   agent_mode TEXT NOT NULL,
                   mode_locked INTEGER NOT NULL
               )"""
        )
        connection.execute(
            "INSERT INTO active_branches VALUES ('chat-1', 'local', '', 'unknown', 1)"
        )

    service = AgentService(database, (profile(),), UnusedSandboxes())  # type: ignore[arg-type]

    assert service.mode_for("chat-1") is AgentMode.READ_ONLY
    assert service.mode_is_locked("chat-1") is True
    with sqlite3.connect(database) as connection:
        stored = connection.execute(
            "SELECT agent_mode FROM active_branches WHERE chat_id = 'chat-1'"
        ).fetchone()[0]
    assert stored == "read_only"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_filesystem_tools"),
    [
        (AgentMode.READ_ONLY, READ_ONLY_FILESYSTEM_TOOLS),
        (AgentMode.EXTENDED, EXTENDED_FILESYSTEM_TOOLS),
    ],
)
async def test_graph_uses_locked_mode_capabilities_and_global_memory_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: AgentMode,
    expected_filesystem_tools: tuple[str, ...],
) -> None:
    captured: dict[str, object] = {}

    class Sandboxes:
        def __init__(self) -> None:
            self.value = StateBackend()
            self.modes: list[AgentMode] = []

        async def backend(self, _chat_id: str, selected: AgentMode):
            self.modes.append(selected)
            return self.value

        def files_dir(self, chat_id: str) -> Path:
            return tmp_path / "sandboxes" / chat_id / "files"

    sandboxes = Sandboxes()
    monkeypatch.setattr(
        agent_service_module, "init_chat_model", lambda *_args, **_kwargs: object()
    )

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        agent_service_module, "create_deep_agent", fake_create_deep_agent
    )
    service = AgentService(
        tmp_path / f"checkpoints-{mode.value}.sqlite3",
        (profile(),),
        sandboxes,  # type: ignore[arg-type]
        global_memory=SQLiteHistory(tmp_path / "history.sqlite3"),
    )

    async def fake_checkpointer():
        return object()

    monkeypatch.setattr(service, "_checkpointer", fake_checkpointer)
    service.set_profile("chat-1", "local")
    service.select_mode("chat-1", mode)
    service.lock_mode("chat-1")

    await service._graph("chat-1")

    filesystem_middleware = captured["middleware"][0]  # type: ignore[index]
    assert tuple(tool.name for tool in filesystem_middleware.tools) == (
        expected_filesystem_tools
    )
    assert {tool.name for tool in captured["tools"]} == {  # type: ignore[union-attr]
        "search_past_chats",
        "read_past_chat",
    }
    assert sandboxes.modes == [mode]
    expected_label = "Read-only" if mode is AgentMode.READ_ONLY else "Extended"
    assert expected_label in captured["system_prompt"]  # type: ignore[operator]
    assert str(sandboxes.files_dir("chat-1")) in captured["system_prompt"]  # type: ignore[operator]


@pytest.mark.asyncio
async def test_deleting_chat_removes_persisted_model_profile(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    service = AgentService(database, (profile(),), UnusedSandboxes())  # type: ignore[arg-type]
    service.set_profile("chat-1", "local")
    await service.delete_chat("chat-1")
    with pytest.raises(RuntimeError, match="being deleted"):
        service.set_profile("chat-1", "local")
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
            return AIMessage(content="Проверка пользователя и версии ядра")

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

    assert title == "Проверка пользователя и версии ядра"
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
async def test_describe_tool_retries_a_first_person_process_title_once(
    tmp_path: Path, monkeypatch
) -> None:
    responses = iter(
        [
            "Изучаю параметры текущей системы",
            "Проверка пользователя и версии ядра",
        ]
    )
    calls = []

    class FakeModel:
        async def ainvoke(self, messages):
            calls.append(messages)
            return AIMessage(content=next(responses))

    monkeypatch.setattr(
        agent_service_module,
        "init_chat_model",
        lambda *_args, **_kwargs: FakeModel(),
    )
    service = AgentService(
        tmp_path / "checkpoints.sqlite3",
        (profile(),),
        UnusedSandboxes(),  # type: ignore[arg-type]
    )
    service.set_profile("chat-1", "local")

    title = await service.describe_tool(
        "chat-1", "execute", '{"command":"whoami; uname -a"}'
    )

    assert title == "Проверка пользователя и версии ядра"
    assert len(calls) == 2
    retry_context = "\n".join(str(message[1]) for message in calls[1])
    assert "Изучаю параметры текущей системы" in retry_context


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


@pytest.mark.asyncio
async def test_describe_chat_summarizes_the_request_with_the_chat_profile(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    class FakeModel:
        async def ainvoke(self, messages):
            calls.append(messages)
            return AIMessage(content="Аудит проекта перед публикацией")

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

    title = await service.describe_chat(
        "chat-1", "Проведи полный аудит проекта и подготовь его к публикации"
    )

    assert title == "Аудит проекта перед публикацией"
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
    assert "короткие русские названия диалогов" in calls[0][0][1]
    assert "<user-request>" in calls[0][1][1]
    assert "полный аудит проекта" in calls[0][1][1]


@pytest.mark.asyncio
async def test_describe_chat_returns_none_for_an_invalid_model_title(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeModel:
        async def ainvoke(self, _messages):
            return AIMessage(content="Короткий заголовок")

    monkeypatch.setattr(
        agent_service_module,
        "init_chat_model",
        lambda *_args, **_kwargs: FakeModel(),
    )
    service = AgentService(
        tmp_path / "checkpoints.sqlite3",
        (profile(),),
        UnusedSandboxes(),  # type: ignore[arg-type]
    )
    service.set_profile("chat-1", "local")

    assert await service.describe_chat("chat-1", "Сделай что-нибудь") is None


@pytest.mark.asyncio
async def test_delete_chat_waits_for_active_agent_run(
    tmp_path: Path, monkeypatch
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class Sandboxes:
        async def backend(self, _chat_id, _mode):
            return object()

        async def push(self, _chat_id, _backend):
            return None

        async def pull(self, _chat_id, _backend):
            return None

    class WaitingGraph:
        async def astream_events(self, *_args, **_kwargs):
            started.set()
            await release.wait()
            yield {
                "event": "on_chain_end",
                "parent_ids": [],
                "data": {"output": {"messages": [AIMessage(content="done")]}},
            }

    service = AgentService(
        tmp_path / "checkpoints.sqlite3",
        (profile(),),
        Sandboxes(),  # type: ignore[arg-type]
    )
    service.set_profile("chat-1", "local")

    async def fake_graph(_chat_id):
        return WaitingGraph()

    monkeypatch.setattr(service, "_graph", fake_graph)
    running = asyncio.create_task(service.run("chat-1", "request"))
    await started.wait()
    deleting = asyncio.create_task(service.delete_chat("chat-1"))
    await asyncio.sleep(0)

    assert deleting.done() is False
    release.set()
    assert await running == "done"
    await deleting
    with pytest.raises(RuntimeError, match="being deleted"):
        await service.run("chat-1", "another request")
    await service.close()


@pytest.mark.asyncio
async def test_failed_first_run_keeps_agent_mode_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Sandboxes:
        async def backend(self, _chat_id: str, _mode: AgentMode):
            return object()

        async def push(self, _chat_id: str, _backend) -> None:
            return None

        async def pull(self, _chat_id: str, _backend) -> None:
            return None

    class FailingGraph:
        async def astream_events(self, *_args, **_kwargs):
            if False:
                yield None
            raise RuntimeError("model failed")

    service = AgentService(
        tmp_path / "checkpoints.sqlite3",
        (profile(),),
        Sandboxes(),  # type: ignore[arg-type]
    )
    service.set_profile("chat-1", "local")
    service.select_mode("chat-1", AgentMode.EXTENDED)

    async def fake_graph(_chat_id: str):
        return FailingGraph()

    monkeypatch.setattr(service, "_graph", fake_graph)

    with pytest.raises(RuntimeError, match="model failed"):
        await service.run("chat-1", "request")

    assert service.mode_for("chat-1") is AgentMode.EXTENDED
    assert service.mode_is_locked("chat-1") is True
    with pytest.raises(ValueError, match="cannot change"):
        service.select_mode("chat-1", AgentMode.READ_ONLY)
