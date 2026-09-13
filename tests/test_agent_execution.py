from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from local_agent_chat.agent_context import count_tokens
from local_agent_chat.agent_events import TextDelta, ToolFinished, ToolStarted
from local_agent_chat.agent_execution import AgentExecution
from local_agent_chat.agent_memory import AgentMemory
from local_agent_chat.chat_bindings import ChatBindings
from local_agent_chat.llm_retry import RetryBlock
from local_agent_chat.runtime import ChatRuntime, Turn
from local_agent_chat.sandbox_files import SandboxFiles
from local_agent_chat.settings import AgentConfig, LLMRetryConfig, ModelProfile
from local_agent_chat.sqlite_history import SQLiteHistory


class Model(FakeMessagesListChatModel):
    requests: list[list[BaseMessage]] = []
    tool_names: list[str] = []

    def bind_tools(self, tools, **kwargs):
        self.tool_names = [t.name for t in tools]
        return self

    def _generate(self, messages, *args, **kwargs):
        self.requests.append(list(messages))
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="accepted"))]
        )


class SummaryModel(Model):
    def _generate(self, messages, *args, **kwargs):
        self.requests.append(list(messages))
        content = "\n".join(str(m.content) for m in messages)
        summary = "Earlier conversation summary."
        if "PIN-ALPHA" in content:
            summary += " The verified project code is PIN-ALPHA."
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=summary))]
        )


def make_runtime(root: Path, *, config=AgentConfig(), model=None, summary=None):
    model = model if model is not None else Model(responses=[])
    summary = summary if summary is not None else SummaryModel(responses=[])
    profile = ModelProfile("test", "Test", "openai:test", "KEY", "test-key")
    bindings = ChatBindings(root / "state.sqlite3", ["test"])
    bindings.open("chat")
    files = SandboxFiles(root / "files", max_file_bytes=10**6, max_chat_bytes=10**7)
    history = SQLiteHistory(root / "history.sqlite3")
    retry = RetryBlock(
        LLMRetryConfig(stream_retries=0),
        lambda *a, **kw: (
            summary if kw.get("max_tokens") == config.summary_tokens else model
        ),
    )
    execution = AgentExecution(
        root / "state.sqlite3",
        (profile,),
        files,
        bindings,
        history,
        config=config,
        retry_block=retry,
    )
    return (
        ChatRuntime(agent=execution, sandbox=files, history=history),
        execution,
        history,
        model,
        summary,
    )


@pytest.mark.parametrize("edited_turn", [1, 2, 25, 26, 50, 75, 99, 100])
async def test_hundred_turn_context_revision_and_restart(tmp_path, edited_turn):
    runtime, execution, history, model, summary = make_runtime(tmp_path)
    for i in range(1, 101):
        await runtime.submit("chat", f"turn-{i}", f"request-{i}")
    assert len(model.requests[-1]) == 200  # system + 99 pairs + new user
    await runtime.revise("chat", f"turn-{edited_turn}", "revised")
    expected = [f"request-{i}" for i in range(1, edited_turn)] + ["revised"]
    assert [m.content for m in model.requests[-1] if m.type == "human"] == expected
    assert (await history.get(f"turn-{edited_turn}")).text == "revised"
    for i in range(edited_turn + 1, 101):
        with pytest.raises(KeyError):
            await history.get(f"turn-{i}")
    await execution.close()
    runtime, execution, _, model, _ = make_runtime(tmp_path)
    await runtime.submit("chat", "continued", "continued")
    assert [m.content for m in model.requests[-1] if m.type == "human"] == expected + [
        "continued"
    ]
    await execution.close()


SMALL_CONTEXT = AgentConfig(
    context_tokens=6000,
    summary_trigger_tokens=1400,
    keep_tokens=350,
    summary_tokens=200,
    max_output_tokens=500,
    max_model_calls=4,
)


async def test_repeated_summaries_survive_restart_and_revision_without_future_context(
    tmp_path,
):
    runtime, execution, history, model, summary = make_runtime(
        tmp_path, config=SMALL_CONTEXT
    )
    snapshots = {}
    for i in range(1, 101):
        snapshots[i] = AgentMemory(tmp_path / "state.sqlite3").load("chat")
        text = (
            f"turn {i}; " + ("PIN-ALPHA; " if i == 1 else "") + "context detail " * 25
        )
        await runtime.submit("chat", f"turn-{i}", text)
    assert len(summary.requests) >= 5
    assert all(
        count_tokens(request)
        < SMALL_CONTEXT.context_tokens - SMALL_CONTEXT.max_output_tokens
        for request in model.requests
    )
    assert "PIN-ALPHA" in str(model.requests[-1])
    assert len(await history.context_messages("chat")) == 200
    before = json.loads((await history.get("turn-50")).memory_checkpoint)
    memory = AgentMemory(tmp_path / "state.sqlite3")
    original_head = memory.load("chat")
    memory.restore("chat", before)
    assert memory.load("chat") == snapshots[50]
    # Restore original head before invoking the real revision transaction.
    memory.save("chat", original_head)
    await runtime.revise("chat", "turn-50", "REVISED-MIDDLE")
    assert "turn 99;" not in str(model.requests[-1])
    assert "REVISED-MIDDLE" in str(model.requests[-1])
    assert "PIN-ALPHA" in str(model.requests[-1])
    await execution.close()
    runtime, execution, history, model, _ = make_runtime(tmp_path, config=SMALL_CONTEXT)
    await runtime.submit("chat", "continued", "Continue using the project code")
    assert "PIN-ALPHA" in str(model.requests[-1])
    assert "REVISED-MIDDLE" in str(model.requests[-1])
    assert "turn 99;" not in str(model.requests[-1])
    await execution.close()


async def test_legacy_chat_import_and_edit_uses_only_visible_preceding_turns(tmp_path):
    runtime, execution, history, model, _ = make_runtime(tmp_path)
    for i in range(1, 6):
        token = json.dumps(
            {
                "version": 3,
                "chat_id": "chat",
                "memory_thread_id": "old-branch",
                "checkpoint_id": str(i),
                "checkpoint_ns": "",
            }
        )
        snapshot = await execution._sandbox.snapshot("chat")
        await history.append(
            Turn(
                id=f"old-{i}",
                chat_id="chat",
                text=f"old request {i}",
                answer=f"old answer {i}",
                memory_checkpoint=token,
                sandbox_snapshot=snapshot,
            )
        )
    await runtime.submit("chat", "new", "new request")
    assert [m.content for m in model.requests[-1] if m.type == "human"] == [
        f"old request {i}" for i in range(1, 6)
    ] + ["new request"]
    await runtime.revise("chat", "old-3", "revised third")
    assert [m.content for m in model.requests[-1] if m.type == "human"] == [
        "old request 1",
        "old request 2",
        "revised third",
    ]
    await execution.close()


class ToolModel(Model):
    def _generate(self, messages, *args, **kwargs):
        self.requests.append(list(messages))
        if messages[-1].type == "tool":
            result = AIMessage(content=str(messages[-1].content))
        else:
            result = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": "/note.txt"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=result)])


async def test_classic_tool_loop_has_only_sandbox_reads_and_emits_events(tmp_path):
    model = ToolModel(responses=[])
    runtime, execution, _, _, _ = make_runtime(tmp_path, model=model)
    execution._sandbox.files_dir("chat").joinpath("note.txt").write_text(
        "sandbox evidence"
    )
    events = []

    async def emit(event):
        events.append(event)

    answer = await runtime.submit("chat", "turn", "Read note.txt", emit)
    assert "sandbox evidence" in answer
    assert set(model.tool_names) == {"ls", "read_file", "glob", "grep"}
    assert any(isinstance(e, ToolStarted) for e in events)
    assert any(isinstance(e, ToolFinished) for e in events)
    assert all(
        not isinstance(e, TextDelta) or "Previous summary" not in e.text for e in events
    )
    await execution.close()


class EndlessToolModel(ToolModel):
    def _generate(self, messages, *args, **kwargs):
        return super()._generate([HumanMessage(content="repeat")], *args, **kwargs)


async def test_summaries_preserve_complete_tool_call_result_groups(tmp_path):
    model = ToolModel(responses=[])
    runtime, execution, _, _, summary = make_runtime(
        tmp_path, model=model, config=SMALL_CONTEXT
    )
    execution._sandbox.files_dir("chat").joinpath("note.txt").write_text(
        "PIN-ALPHA verified file evidence " * 10
    )
    for index in range(25):
        await runtime.submit("chat", f"turn-{index}", "Read note.txt. " * 20)
    assert len(summary.requests) >= 3
    for request in model.requests:
        pending = set()
        for message in request:
            if isinstance(message, AIMessage):
                assert not pending
                pending = {call["id"] for call in message.tool_calls}
            elif message.type == "tool":
                assert message.tool_call_id in pending
                pending.remove(message.tool_call_id)
            else:
                assert not pending
        assert not pending
        assert (
            count_tokens(request) + SMALL_CONTEXT.max_output_tokens
            < SMALL_CONTEXT.context_tokens
        )
    await execution.close()


async def test_model_call_limit_stops_tool_loop_and_keeps_previous_memory(tmp_path):
    runtime, execution, history, _, _ = make_runtime(
        tmp_path, model=EndlessToolModel(responses=[]), config=SMALL_CONTEXT
    )
    before = await execution.checkpoint("chat")
    with pytest.raises(Exception, match="limit|Limit"):
        await runtime.submit("chat", "loop", "repeat tools")
    assert execution._memory.load("chat") == []
    assert not await history.has_chat("chat")
    await execution.restore("chat", before)
    await execution.close()


async def test_checkpoint_cannot_restore_another_chat(tmp_path):
    _, execution, _, _, _ = make_runtime(tmp_path)
    token = await execution.checkpoint("chat")
    execution._bindings.open("other")
    with pytest.raises(ValueError, match="another Chat"):
        await execution.restore("other", token)
    await execution.close()


async def test_summary_failure_and_oversized_request_leave_history_intact(tmp_path):
    runtime, execution, history, model, summary = make_runtime(
        tmp_path, config=SMALL_CONTEXT
    )
    await runtime.submit("chat", "first", "PIN-ALPHA")
    before = execution._memory.load("chat")
    with pytest.raises(ValueError, match="лимит"):
        await runtime.submit("chat", "huge", "x" * 30000)
    assert execution._memory.load("chat") == before
    assert len(await history.context_messages("chat")) == 2
    await execution.close()


@pytest.mark.parametrize("failure", ["provider", "empty", "cancel", "truncated"])
async def test_summary_failure_or_cancel_restores_exact_previous_context(
    tmp_path, monkeypatch, failure
):
    import asyncio

    runtime, execution, history, model, summary = make_runtime(
        tmp_path, config=SMALL_CONTEXT
    )
    # A large seeded old context forces summary in the very next turn.
    original = [
        HumanMessage(content="PIN-ALPHA " + "old detail " * 400),
        AIMessage(content="acknowledged"),
    ]
    execution._memory.save("chat", original)
    started = asyncio.Event()

    async def fail_summary(_self, _messages, *args, **kwargs):
        started.set()
        if failure == "cancel":
            await asyncio.Event().wait()
        if failure == "provider":
            raise RuntimeError("summary provider failed")
        if failure == "truncated":
            return AIMessage(
                content="partial fact", response_metadata={"finish_reason": "length"}
            )
        return AIMessage(content="")

    monkeypatch.setattr(SummaryModel, "ainvoke", fail_summary)
    task = asyncio.create_task(runtime.submit("chat", "new", "Continue"))
    await asyncio.wait_for(started.wait(), 3)
    if failure == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(RuntimeError, match="summary"):
            await task
    assert execution._memory.load("chat") == original
    assert not await history.has_chat("chat")
    assert not model.requests
    await execution.close()


async def test_empty_main_answer_does_not_commit_a_turn(tmp_path, monkeypatch):
    runtime, execution, history, _, _ = make_runtime(tmp_path)
    await runtime.submit("chat", "first", "Keep this fact")
    before = execution._memory.load("chat")
    monkeypatch.setattr(
        Model,
        "_generate",
        lambda *a, **kw: ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=" "))]
        ),
    )
    with pytest.raises(RuntimeError, match="empty answer"):
        await runtime.submit("chat", "empty", "Continue")
    assert execution._memory.load("chat") == before
    assert len(await history.context_messages("chat")) == 2
    await execution.close()


async def test_summary_processes_every_chunk_of_large_legacy_context(tmp_path):
    from local_agent_chat.agent_context import ContextSummary

    model = SummaryModel(responses=[])
    retry = RetryBlock(LLMRetryConfig(stream_retries=0), lambda *a, **kw: model)
    middleware = ContextSummary(model, SMALL_CONTEXT, retry)
    original = [
        HumanMessage(content=("early marker " + "filler " * 10000 + " late marker"))
    ]
    await middleware._acreate_summary(original)
    assert len(model.requests) > 5
    fragments = "".join(
        str(request[-1].content).split("Next conversation fragment:\n", 1)[1]
        for request in model.requests
    )
    assert "early marker" in fragments and "late marker" in fragments
    assert "filler " * 10000 in fragments
    assert all(
        count_tokens(request) + SMALL_CONTEXT.summary_tokens
        < SMALL_CONTEXT.context_tokens
        for request in model.requests
    )
