import asyncio
import json
from pathlib import Path
from typing import Annotated, NotRequired, TypedDict

import pytest
from deepagents.backends import StateBackend
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import StreamChunkTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from local_agent_chat import deep_agent_execution as agent_execution_module
from local_agent_chat.agent_modes import READ_FILESYSTEM_TOOLS, AgentMode
from local_agent_chat.chat_bindings import ChatBindings
from local_agent_chat.deep_agent_execution import DeepAgentExecution
from local_agent_chat.long_term_memory import MarkdownMemory
from local_agent_chat.settings import LLMRetryConfig, ModelProfile
from local_agent_chat.sqlite_history import SQLiteHistory


class UnusedSandboxes:
    async def backend(self, chat_id: str):
        raise AssertionError("backend should not be created while selecting a profile")


def profile() -> ModelProfile:
    return ModelProfile("local", "Local", "openai:test", "TEST_KEY", "key")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_filesystem_tools"),
    [
        (AgentMode.CHAT_FILES, READ_FILESYSTEM_TOOLS),
        (AgentMode.HOST_FILES, READ_FILESYSTEM_TOOLS),
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
    fake_model = FakeListChatModel(responses=["unused"])
    monkeypatch.setattr(
        agent_execution_module,
        "init_chat_model",
        lambda *_args, **_kwargs: fake_model,
    )

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        agent_execution_module, "create_deep_agent", fake_create_deep_agent
    )
    (tmp_path / "skills").mkdir()
    database = tmp_path / f"checkpoints-{mode.value}.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")
    bindings.select_mode("chat-1", mode)
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile(),),
        sandboxes,  # type: ignore[arg-type]
        global_memory=SQLiteHistory(tmp_path / "history.sqlite3"),
        long_term_memory=MarkdownMemory(tmp_path / "memory" / "MEMORY.md"),
        skills_dir=tmp_path / "skills",
        chat_bindings=bindings,
    )

    async def fake_checkpointer():
        return object()

    monkeypatch.setattr(execution, "_checkpointer", fake_checkpointer)

    await execution._graph("chat-1")

    filesystem_middleware = captured["middleware"][0]  # type: ignore[index]
    assert tuple(tool.name for tool in filesystem_middleware.tools) == (
        expected_filesystem_tools
    )
    summarization_middleware = captured["middleware"][1]  # type: ignore[index]
    assert summarization_middleware.name == "SummarizationMiddleware"
    assert summarization_middleware._lc_helper._summary_model is captured["model"]
    memory_middleware = captured["middleware"][2]  # type: ignore[index]
    assert memory_middleware.name == "_RefreshingMemoryMiddleware"
    assert "remember_context" in memory_middleware.system_prompt
    assert {tool.name for tool in memory_middleware.tools} == {
        "remember_context",
        "forget_context",
    }
    assert {tool.name for tool in captured["tools"]} == {  # type: ignore[union-attr]
        "search_past_chats",
        "read_past_chat",
    }
    subagent = captured["subagents"][0]  # type: ignore[index]
    assert subagent["name"] == "general-purpose"
    assert "without its mutation tools" in subagent["description"]
    subagent_filesystem = subagent["middleware"][0]
    assert tuple(tool.name for tool in subagent_filesystem.tools) == (
        READ_FILESYSTEM_TOOLS
    )
    subagent_memory = subagent["middleware"][2]
    assert subagent_memory.name == "_RefreshingMemoryMiddleware"
    assert subagent_memory.tools == ()
    assert "has no `remember_context` or `forget_context` tool" in (
        subagent_memory.system_prompt
    )
    assert subagent["skills"] == [(tmp_path / "skills").resolve().as_posix()]
    assert sandboxes.modes == [mode]
    expected_label = (
        "Chat Files Agent Mode"
        if mode is AgentMode.CHAT_FILES
        else "Host Files Agent Mode"
    )
    assert expected_label in captured["system_prompt"]  # type: ignore[operator]
    prompt = captured["system_prompt"]
    assert "helpful assistant operating inside the LocalChat chat harness" in prompt
    assert "concise, neutral and matter-of-fact language" in prompt
    assert "Do not add filler" in prompt
    assert all(term in prompt for term in ("emojis", "emoticons", "decorative symbols"))
    if mode is AgentMode.HOST_FILES:
        assert str(sandboxes.files_dir("chat-1")) in prompt  # type: ignore[operator]
    else:
        assert str(sandboxes.files_dir("chat-1")) not in prompt  # type: ignore[operator]
    assert captured["skills"] == [(tmp_path / "skills").resolve().as_posix()]


def test_rejects_a_missing_agent_skills_directory(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")

    with pytest.raises(ValueError, match="Agent skills directory does not exist"):
        DeepAgentExecution(
            database,
            (profile(),),
            UnusedSandboxes(),  # type: ignore[arg-type]
            skills_dir=tmp_path / "missing-skills",
            chat_bindings=bindings,
        )


@pytest.mark.asyncio
async def test_main_model_uses_configured_llm_retry_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_calls: list[tuple[str, dict[str, object]]] = []
    captured: dict[str, object] = {}

    class FakeModel(FakeListChatModel):
        def bind_tools(self, _tools, **_kwargs):
            return self

    class Sandboxes:
        def __init__(self) -> None:
            self.value = StateBackend()

        async def backend(self, _chat_id: str, _mode: AgentMode):
            return self.value

        def files_dir(self, chat_id: str) -> Path:
            return tmp_path / "sandboxes" / chat_id / "files"

    def fake_init(model: str, **kwargs):
        init_calls.append((model, kwargs))
        return FakeModel(responses=["Надёжная генерация названия диалога"])

    monkeypatch.setattr(agent_execution_module, "init_chat_model", fake_init)

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        agent_execution_module, "create_deep_agent", fake_create_deep_agent
    )
    configured_profile = ModelProfile(
        "local",
        "Local",
        "openai:test",
        "TEST_KEY",
        "key",
        base_url="https://models.example.test/v1",
    )
    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (configured_profile,),
        Sandboxes(),  # type: ignore[arg-type]
        llm_retry=LLMRetryConfig(
            max_retries=7,
            request_timeout_seconds=17.0,
            stream_chunk_timeout_seconds=19.0,
            auxiliary_timeout_seconds=23.0,
        ),
        chat_bindings=bindings,
    )

    async def fake_checkpointer():
        return object()

    monkeypatch.setattr(execution, "_checkpointer", fake_checkpointer)

    await execution._graph("chat-1")

    shared_policy = {
        "api_key": "key",
        "base_url": "https://models.example.test/v1",
        "max_retries": 7,
        "stream_chunk_timeout": 19.0,
        "timeout": 17.0,
    }
    assert init_calls == [("openai:test", shared_policy)]
    subagent = captured["subagents"][0]  # type: ignore[index]
    assert tuple(tool.name for tool in subagent["middleware"][0].tools) == (
        READ_FILESYSTEM_TOOLS
    )


@pytest.mark.asyncio
async def test_deleting_chat_removes_persisted_model_profile(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile(),),
        UnusedSandboxes(),  # type: ignore[arg-type]
        chat_bindings=bindings,
    )

    await execution.delete_chat("chat-1")
    with pytest.raises(RuntimeError, match="being deleted"):
        bindings.open("chat-1", "local")
    await execution.close()

    reopened_bindings = ChatBindings(database, ("local",))
    assert reopened_bindings.get("chat-1") is None


@pytest.mark.asyncio
async def test_execution_requires_chat_configuration_to_lock_the_mode(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")
    execution = DeepAgentExecution(
        database,
        (profile(),),
        UnusedSandboxes(),  # type: ignore[arg-type]
        chat_bindings=bindings,
    )

    with pytest.raises(RuntimeError, match="locked before execution"):
        await execution.run("chat-1", "request")


@pytest.mark.asyncio
async def test_checkpoint_is_chat_bound_and_legacy_tokens_remain_readable(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")
    bindings.open("chat-2", "local")
    execution = DeepAgentExecution(
        database,
        (profile(),),
        UnusedSandboxes(),  # type: ignore[arg-type]
        chat_bindings=bindings,
    )

    checkpoint = await execution.checkpoint("chat-1")

    assert json.loads(checkpoint) == {
        "version": 3,
        "chat_id": "chat-1",
        "memory_thread_id": "chat-1",
        "checkpoint_ns": "",
        "checkpoint_id": None,
    }
    with pytest.raises(ValueError, match="another Chat"):
        await execution.restore("chat-2", checkpoint)

    legacy = json.dumps({"checkpoint_ns": "", "checkpoint_id": None})
    await execution.restore("chat-2", legacy)
    restored = bindings.get("chat-2")
    assert restored is not None
    assert restored.memory_thread_id.startswith("chat-2:")
    await execution.close()


@pytest.mark.asyncio
async def test_restoring_an_empty_checkpoint_starts_clean_memory_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class State(TypedDict):
        messages: Annotated[list[object], add_messages]

    class Sandboxes:
        async def backend(self, _chat_id: str, _mode: AgentMode) -> object:
            return object()

    def model(state: State) -> State:
        requests = [
            str(message.content)
            for message in state["messages"]
            if isinstance(message, HumanMessage)
        ]
        return {"messages": [AIMessage(content=f"seen:{'|'.join(requests)}")]}

    builder = StateGraph(State)
    builder.add_node("model", model)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile(),),
        Sandboxes(),  # type: ignore[arg-type]
        chat_bindings=bindings,
    )
    graph = builder.compile(checkpointer=await execution._checkpointer())

    async def fake_graph(_chat_id: str):
        return graph

    monkeypatch.setattr(execution, "_graph", fake_graph)
    empty_checkpoint = await execution.checkpoint("chat-1")

    assert await execution.run("chat-1", "original") == "seen:original"
    await execution.restore("chat-1", empty_checkpoint)
    clean_thread_id = bindings.get("chat-1").memory_thread_id  # type: ignore[union-attr]
    assert clean_thread_id.startswith("chat-1:")
    assert await execution.run("chat-1", "revised") == "seen:revised"
    await execution.close()

    reopened_bindings = ChatBindings(database, ("local",))
    reopened = DeepAgentExecution(
        database,
        (profile(),),
        Sandboxes(),  # type: ignore[arg-type]
        chat_bindings=reopened_bindings,
    )
    graph = builder.compile(checkpointer=await reopened._checkpointer())
    monkeypatch.setattr(reopened, "_graph", fake_graph)

    assert reopened_bindings.get("chat-1").memory_thread_id == clean_thread_id  # type: ignore[union-attr]
    assert await reopened.run("chat-1", "continuation") == ("seen:revised|continuation")
    await reopened.close()


@pytest.mark.asyncio
async def test_restored_checkpoint_is_a_durable_head_not_a_volatile_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class State(TypedDict):
        messages: Annotated[list[object], add_messages]

    class Sandboxes:
        async def backend(self, _chat_id: str, _mode: AgentMode) -> object:
            return object()

    def model(state: State) -> State:
        requests = [
            str(message.content)
            for message in state["messages"]
            if isinstance(message, HumanMessage)
        ]
        if requests[-1] == "failed":
            raise RuntimeError("model failed")
        return {"messages": [AIMessage(content=f"seen:{'|'.join(requests)}")]}

    builder = StateGraph(State)
    builder.add_node("model", model)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile(),),
        Sandboxes(),  # type: ignore[arg-type]
        chat_bindings=bindings,
    )
    graph = builder.compile(checkpointer=await execution._checkpointer())

    async def fake_graph(_chat_id: str):
        return graph

    monkeypatch.setattr(execution, "_graph", fake_graph)

    assert await execution.run("chat-1", "original") == "seen:original"
    source_checkpoint = await execution.checkpoint("chat-1")
    source = json.loads(source_checkpoint)
    with pytest.raises(RuntimeError, match="model failed"):
        await execution.run("chat-1", "failed")

    await execution.restore("chat-1", source_checkpoint)
    materialized = json.loads(await execution.checkpoint("chat-1"))
    assert materialized["checkpoint_id"] == source["checkpoint_id"]
    assert materialized["memory_thread_id"] != source["memory_thread_id"]
    await execution.close()

    reopened_bindings = ChatBindings(database, ("local",))
    reopened = DeepAgentExecution(
        database,
        (profile(),),
        Sandboxes(),  # type: ignore[arg-type]
        chat_bindings=reopened_bindings,
    )
    graph = builder.compile(checkpointer=await reopened._checkpointer())
    monkeypatch.setattr(reopened, "_graph", fake_graph)

    assert await reopened.run("chat-1", "continuation") == (
        "seen:original|continuation"
    )
    await reopened.close()


@pytest.mark.asyncio
async def test_concurrent_chats_share_one_checkpointer_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")
    bindings.open("chat-2", "local")
    connect_calls = 0
    close_calls = 0

    class FakeConnection:
        async def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    class FakeSaver:
        def __init__(self, _connection: FakeConnection) -> None:
            pass

        async def setup(self) -> None:
            await asyncio.sleep(0)

        async def aget_tuple(self, _config):
            return None

    async def fake_connect(_database: Path) -> FakeConnection:
        nonlocal connect_calls
        connect_calls += 1
        await asyncio.sleep(0)
        return FakeConnection()

    monkeypatch.setattr(agent_execution_module.aiosqlite, "connect", fake_connect)
    monkeypatch.setattr(agent_execution_module, "AsyncSqliteSaver", FakeSaver)
    execution = DeepAgentExecution(
        database,
        (profile(),),
        UnusedSandboxes(),  # type: ignore[arg-type]
        chat_bindings=bindings,
    )

    checkpoints = await asyncio.gather(
        execution.checkpoint("chat-1"), execution.checkpoint("chat-2")
    )

    assert connect_calls == 1
    assert {json.loads(token)["chat_id"] for token in checkpoints} == {
        "chat-1",
        "chat-2",
    }
    await execution.close()
    assert close_calls == 1


@pytest.mark.asyncio
async def test_delete_chat_waits_for_active_agent_run(
    tmp_path: Path, monkeypatch
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class Sandboxes:
        async def backend(self, _chat_id, _mode):
            return object()

    class WaitingGraph:
        async def astream_events(self, *_args, **_kwargs):
            started.set()
            await release.wait()
            yield {
                "event": "on_chain_end",
                "parent_ids": [],
                "data": {"output": {"messages": [AIMessage(content="done")]}},
            }

    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile(),),
        Sandboxes(),  # type: ignore[arg-type]
        chat_bindings=bindings,
    )

    async def fake_graph(_chat_id):
        return WaitingGraph()

    monkeypatch.setattr(execution, "_graph", fake_graph)
    checkpoint = await execution.checkpoint("chat-1")
    running = asyncio.create_task(execution.run("chat-1", "request"))
    await started.wait()
    deleting = asyncio.create_task(execution.delete_chat("chat-1"))
    await asyncio.sleep(0)

    assert deleting.done() is False
    with pytest.raises(RuntimeError, match="being deleted"):
        await execution.checkpoint("chat-1")
    with pytest.raises(RuntimeError, match="being deleted"):
        await execution.restore("chat-1", checkpoint)
    release.set()
    assert await running == "done"
    await deleting
    with pytest.raises(RuntimeError, match="being deleted"):
        await execution.run("chat-1", "another request")
    await execution.close()


@pytest.mark.asyncio
async def test_failed_run_keeps_the_prelocked_agent_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Sandboxes:
        async def backend(self, _chat_id: str, _mode: AgentMode):
            return object()

    class FailingGraph:
        async def astream_events(self, *_args, **_kwargs):
            if False:
                yield None
            raise RuntimeError("model failed")

    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")
    bindings.select_mode("chat-1", AgentMode.HOST_FILES)
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile(),),
        Sandboxes(),  # type: ignore[arg-type]
        chat_bindings=bindings,
    )

    async def fake_graph(_chat_id: str):
        return FailingGraph()

    monkeypatch.setattr(execution, "_graph", fake_graph)

    with pytest.raises(RuntimeError, match="model failed"):
        await execution.run("chat-1", "request")

    binding = bindings.get("chat-1")
    assert binding is not None
    assert binding.mode is AgentMode.HOST_FILES
    assert binding.mode_locked is True
    with pytest.raises(ValueError, match="cannot change"):
        bindings.select_mode("chat-1", AgentMode.CHAT_FILES)


@pytest.mark.asyncio
async def test_run_allows_a_finite_graph_beyond_langgraph_default_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class State(TypedDict):
        messages: list[object]
        count: NotRequired[int]

    def step(state: State) -> State:
        count = state.get("count", 0) + 1
        messages: list[object] = (
            [AIMessage(content="done")] if count == 26 else state["messages"]
        )
        return {"count": count, "messages": messages}

    def route(state: State) -> str:
        return END if state["count"] >= 26 else "step"

    builder = StateGraph(State)
    builder.add_node("step", step)
    builder.add_edge(START, "step")
    builder.add_conditional_edges("step", route)

    class Sandboxes:
        async def backend(self, _chat_id: str, _mode: AgentMode) -> object:
            return object()

    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile(),),
        Sandboxes(),  # type: ignore[arg-type]
        chat_bindings=bindings,
    )
    graph = builder.compile(checkpointer=await execution._checkpointer())

    async def fake_graph(_chat_id: str):
        return graph

    monkeypatch.setattr(execution, "_graph", fake_graph)

    assert await execution.run("chat-1", "ordinary analysis") == "done"
    await execution.close()


@pytest.mark.asyncio
async def test_run_never_replays_the_graph_after_an_empty_stream_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Sandboxes:
        async def backend(self, _chat_id: str, _mode: AgentMode) -> object:
            return object()

    class State(TypedDict):
        messages: list[object]
        tool_runs: NotRequired[int]

    calls = {"tool": 0, "model": 0}
    checkpoint_writes = 0
    model_overlapped_checkpoint = False

    def tool(state: State) -> State:
        calls["tool"] += 1
        return {"messages": state["messages"], "tool_runs": calls["tool"]}

    def model(_state: State) -> State:
        nonlocal model_overlapped_checkpoint
        calls["model"] += 1
        model_overlapped_checkpoint = (
            model_overlapped_checkpoint or checkpoint_writes > 0
        )
        if calls["model"] == 1:
            raise StreamChunkTimeoutError(0.01, model_name="fake", chunks_received=0)
        return {"messages": [AIMessage(content="recovered")]}

    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile(),),
        Sandboxes(),  # type: ignore[arg-type]
        llm_retry=LLMRetryConfig(max_retries=1, stream_retries=1),
        chat_bindings=bindings,
    )

    builder = StateGraph(State)
    builder.add_node("tool", tool)
    builder.add_node("model", model)
    builder.add_edge(START, "tool")
    builder.add_edge("tool", "model")
    builder.add_edge("model", END)
    saver = await execution._checkpointer()
    original_aput = type(saver).aput

    async def delayed_aput(self, *args, **kwargs):
        nonlocal checkpoint_writes
        checkpoint_writes += 1
        try:
            await asyncio.sleep(0.01)
            return await original_aput(self, *args, **kwargs)
        finally:
            checkpoint_writes -= 1

    monkeypatch.setattr(type(saver), "aput", delayed_aput)
    graph = builder.compile(checkpointer=saver)

    async def fake_graph(_chat_id: str):
        return graph

    monkeypatch.setattr(execution, "_graph", fake_graph)

    with pytest.raises(StreamChunkTimeoutError):
        await execution.run("chat-1", "read two files")

    assert calls == {"tool": 1, "model": 1}
    assert model_overlapped_checkpoint is False
    await execution.close()


@pytest.mark.asyncio
async def test_stream_timeout_does_not_replay_a_materialized_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class State(TypedDict):
        messages: list[object]

    class Sandboxes:
        async def backend(self, _chat_id: str, _mode: AgentMode) -> object:
            return object()

    calls = {"tool": 0, "model": 0}
    revising = False
    revised_model_calls = 0

    def tool(state: State) -> State:
        calls["tool"] += 1
        return {"messages": state["messages"]}

    def model(_state: State) -> State:
        nonlocal revised_model_calls
        calls["model"] += 1
        if revising:
            revised_model_calls += 1
            if revised_model_calls == 1:
                raise StreamChunkTimeoutError(
                    0.01,
                    model_name="fake",
                    chunks_received=0,
                )
            return {"messages": [AIMessage(content="recovered revision")]}
        return {"messages": [AIMessage(content="original answer")]}

    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile(),),
        Sandboxes(),  # type: ignore[arg-type]
        llm_retry=LLMRetryConfig(max_retries=1, stream_retries=1),
        chat_bindings=bindings,
    )

    builder = StateGraph(State)
    builder.add_node("tool", tool)
    builder.add_node("model", model)
    builder.add_edge(START, "tool")
    builder.add_edge("tool", "model")
    builder.add_edge("model", END)
    delegate = builder.compile(checkpointer=await execution._checkpointer())
    attempts: list[tuple[object, dict[str, object], dict[str, object]]] = []

    class RecordingGraph:
        async def astream_events(self, graph_input, config, **kwargs):
            recorded_config = {
                **config,
                "configurable": dict(config["configurable"]),
            }
            attempts.append((graph_input, recorded_config, dict(kwargs)))
            async for event in delegate.astream_events(
                graph_input,
                config=config,
                **kwargs,
            ):
                yield event

    graph = RecordingGraph()

    async def fake_graph(_chat_id: str):
        return graph

    monkeypatch.setattr(execution, "_graph", fake_graph)

    assert await execution.run("chat-1", "original") == "original answer"
    checkpoint = await execution.checkpoint("chat-1")
    source = json.loads(checkpoint)
    assert source["checkpoint_id"]
    await execution.restore("chat-1", checkpoint)
    restored_thread_id = bindings.get("chat-1").memory_thread_id  # type: ignore[union-attr]
    assert restored_thread_id != source["memory_thread_id"]
    revising = True

    with pytest.raises(StreamChunkTimeoutError):
        await execution.run("chat-1", "revised")

    assert calls == {"tool": 2, "model": 2}
    assert revised_model_calls == 1
    assert attempts[-1][0] is not None
    assert "checkpoint_id" not in attempts[-1][1]["configurable"]
    assert attempts[-1][1]["configurable"]["thread_id"] == restored_thread_id
    assert attempts[-1][2]["durability"] == "sync"
    await execution.close()


@pytest.mark.asyncio
async def test_run_propagates_timeout_with_an_open_tool_without_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Sandboxes:
        async def backend(self, _chat_id: str, _mode: AgentMode) -> object:
            return object()

    class StalledToolGraph:
        def __init__(self) -> None:
            self.calls = 0

        async def astream_events(self, *_args, **_kwargs):
            self.calls += 1
            yield {
                "event": "on_tool_start",
                "run_id": "task-1",
                "name": "task",
                "data": {"input": {"description": "delegate work"}},
            }
            raise StreamChunkTimeoutError(
                0.01,
                model_name="fake",
                chunks_received=0,
            )

    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile(),),
        Sandboxes(),  # type: ignore[arg-type]
        llm_retry=LLMRetryConfig(max_retries=3, stream_retries=3),
        chat_bindings=bindings,
    )
    graph = StalledToolGraph()

    async def fake_graph(_chat_id: str):
        return graph

    monkeypatch.setattr(execution, "_graph", fake_graph)

    with pytest.raises(StreamChunkTimeoutError):
        await execution.run("chat-1", "delegate this")

    assert graph.calls == 1


@pytest.mark.asyncio
async def test_run_propagates_timeout_after_a_tool_error_without_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Sandboxes:
        async def backend(self, _chat_id: str, _mode: AgentMode) -> object:
            return object()

    class FailedToolGraph:
        def __init__(self) -> None:
            self.calls = 0

        async def astream_events(self, *_args, **_kwargs):
            self.calls += 1
            yield {
                "event": "on_tool_start",
                "run_id": "task-1",
                "name": "task",
                "data": {"input": {}},
            }
            error = StreamChunkTimeoutError(
                0.01,
                model_name="fake",
                chunks_received=0,
            )
            yield {
                "event": "on_tool_error",
                "run_id": "task-1",
                "name": "task",
                "data": {"error": error},
            }
            raise error

    database = tmp_path / "checkpoints.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")
    bindings.lock("chat-1")
    execution = DeepAgentExecution(
        database,
        (profile(),),
        Sandboxes(),  # type: ignore[arg-type]
        llm_retry=LLMRetryConfig(max_retries=3, stream_retries=3),
        chat_bindings=bindings,
    )
    graph = FailedToolGraph()

    async def fake_graph(_chat_id: str):
        return graph

    monkeypatch.setattr(execution, "_graph", fake_graph)

    with pytest.raises(StreamChunkTimeoutError):
        await execution.run("chat-1", "delegate this")

    assert graph.calls == 1
