from __future__ import annotations

import asyncio
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import chainlit as cl
import chainlit.data as chainlit_data
import chainlit.socket as chainlit_socket
import pytest
from chainlit.config import config
from chainlit.context import ChainlitContext, context_var
from chainlit.emitter import BaseChainlitEmitter
from chainlit.server import sio
from chainlit.session import HTTPSession, WebsocketSession
from chainlit.user import User
from chainlit.utils import utc_now

from local_agent_chat.agent_events import TextDelta
from local_agent_chat.chainlit_ui import ChainlitTurnView
from local_agent_chat.runtime import ChatRuntime


class HistoryAgent:
    def __init__(self):
        self.messages = []
        self.started = asyncio.Event()

    async def checkpoint(self, _chat_id):
        return json.dumps(self.messages)

    async def restore(self, _chat_id, token):
        self.messages = json.loads(token)

    async def run(self, _chat_id, text, _emit=None):
        self.messages.append(text)
        if text == "fail":
            raise RuntimeError("provider failed")
        if text == "wait":
            self.started.set()
            await asyncio.Event().wait()
        return "seen:" + "|".join(self.messages)


class TimelineEmitter(BaseChainlitEmitter):
    timeline = None

    def __init__(self, session):
        super().__init__(session)
        self.toasts = []

    async def send_toast(self, message, style="info"):
        self.toasts.append((message, style))

    def set_chat_settings(self, settings):
        self.session.chat_settings = settings

    async def resume_thread(self, thread):
        self.timeline = thread


@pytest.fixture
async def chat(tmp_path, monkeypatch, request):
    monkeypatch.setattr(config.code, "on_stop", None)
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv(
        "MODEL_PROFILES_FILE", str(Path("models.example.yaml").resolve())
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # App imports install global Chainlit handlers. Restore them after this test.
    monkeypatch.setitem(
        sio.handlers["/"], "edit_message", sio.handlers["/"]["edit_message"]
    )
    spec = importlib.util.spec_from_file_location(
        "_revision_lifecycle_app", Path("app.py")
    )
    app = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = app
    spec.loader.exec_module(app)
    layer = app.chainlit_layer
    monkeypatch.setattr(chainlit_data, "_data_layer", layer)
    monkeypatch.setattr(chainlit_data, "_data_layer_initialized", True)
    monkeypatch.setattr(app, "_start_chat_title", lambda *_args: None)
    agent = HistoryAgent()
    app.runtime = ChatRuntime(
        agent=agent, sandbox=app.sandbox_files, history=app.runtime_history
    )
    user = await layer.create_user(User(identifier="local-user", metadata={}))
    session = HTTPSession(
        id="revision-session", client_type="webapp", thread_id="chat-1", user=user
    )
    emitter = TimelineEmitter(session)
    token = context_var.set(ChainlitContext(session, emitter))
    try:
        await app.on_chat_start()
        await layer.update_thread("chat-1", user_id=user.id)
        for i in range(1, getattr(request, "param", 5) + 1):
            message = cl.Message(
                id=f"turn-{i}",
                type="user_message",
                content=f"request-{i}",
                created_at=utc_now(),
            )
            await layer.create_step(message.to_dict())
            await app.on_message(message)
        yield app, agent, emitter
    finally:
        context_var.reset(token)
        await app.agent_execution.close()
        await layer.close()
        sys.modules.pop(spec.name, None)


def active_rows(app):
    with sqlite3.connect(
        app.settings.data_dir / "runtime-history.sqlite3"
    ) as connection:
        return connection.execute(
            "SELECT id, text, answer FROM turns ORDER BY sequence"
        ).fetchall()


def user_outputs(thread):
    return [
        step["output"] for step in thread["steps"] if step["type"] == "user_message"
    ]


async def edit(app, text, turn_id="turn-3"):
    await app.on_edit_message({"message": {"id": turn_id, "output": text}})


async def test_edit_replaces_third_turn_and_syncs_live_timeline(chat):
    app, agent, emitter = chat
    await edit(app, "revised-3")
    assert [row[:2] for row in active_rows(app)] == [
        ("turn-1", "request-1"),
        ("turn-2", "request-2"),
        ("turn-3", "revised-3"),
    ]
    assert agent.messages == ["request-1", "request-2", "revised-3"]
    assert user_outputs(emitter.timeline) == agent.messages
    assert sum(s["type"] == "assistant_message" for s in emitter.timeline["steps"]) == 3
    await edit(app, "revised-2", "turn-2")
    assert user_outputs(emitter.timeline) == ["request-1", "revised-2"]
    assert [row[1] for row in active_rows(app)] == ["request-1", "revised-2"]
    with sqlite3.connect(app.settings.data_dir / "chainlit.sqlite3") as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM step_revisions").fetchone()[0] == 0
        )


async def test_unchanged_edit_restores_optimistically_truncated_ui(chat):
    app, agent, emitter = chat
    before = active_rows(app)
    await edit(app, "request-3")
    assert active_rows(app) == before
    assert user_outputs(emitter.timeline) == [f"request-{i}" for i in range(1, 6)]
    assert agent.messages == [f"request-{i}" for i in range(1, 6)]


@pytest.mark.parametrize("failure", ["provider", "cancel", "persistence"])
async def test_failed_edit_restores_sqlite_memory_and_live_timeline(
    chat, monkeypatch, caplog, failure
):
    app, agent, emitter = chat
    before = active_rows(app)
    if failure == "provider":
        await edit(app, "fail")
        assert emitter.toasts and emitter.toasts[-1][1] == "error"
        assert "восстановлена" in emitter.toasts[-1][0]
        assert "provider failed" in caplog.text
    elif failure == "cancel":
        task = asyncio.create_task(edit(app, "wait"))
        await asyncio.wait_for(agent.started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        original_upsert = app.chainlit_layer._upsert_step

        async def fail_answer(session, step):
            if step.get("output") == "seen:request-1|request-2|revised-3":
                await asyncio.sleep(0.02)
                raise RuntimeError("answer write failed")
            await original_upsert(session, step)

        monkeypatch.setattr(app.chainlit_layer, "_upsert_step", fail_answer)
        with pytest.raises(RuntimeError, match="answer write failed"):
            await edit(app, "revised-3")
    assert active_rows(app) == before
    assert agent.messages == [f"request-{i}" for i in range(1, 6)]
    assert user_outputs(emitter.timeline) == agent.messages
    with sqlite3.connect(app.settings.data_dir / "chainlit.sqlite3") as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM step_revisions").fetchone()[0] == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM steps WHERE type = 'assistant_message'"
            ).fetchone()[0]
            == 5
        )


async def test_edit_rejects_a_message_from_another_chat(chat):
    app, agent, emitter = chat
    await app.chainlit_layer.create_step(
        {
            "id": "foreign",
            "threadId": "chat-2",
            "type": "user_message",
            "output": "foreign",
            "createdAt": "2026-09-12T00:00:00Z",
        }
    )
    with pytest.raises(ValueError, match="does not belong"):
        await edit(app, "changed", "foreign")
    assert (await app.chainlit_layer.get_step("foreign"))["output"] == "foreign"
    assert len(active_rows(app)) == 5


async def test_socket_edit_registers_the_task_stopped_by_chainlit(chat, monkeypatch):
    app, agent, emitter = chat
    session = emitter.session
    monkeypatch.setattr(WebsocketSession, "require", lambda _sid: session)
    import local_agent_chat.chainlit_revision as adapter

    monkeypatch.setattr(adapter, "init_ws_context", lambda _session: context_var.get())
    await sio.handlers["/"]["edit_message"](
        "socket", {"message": {"id": "turn-3", "output": "wait"}}
    )
    await asyncio.wait_for(agent.started.wait(), 2)
    assert session.current_task is not None
    session.current_task.cancel()
    await session.current_task
    assert len(active_rows(app)) == 5
    assert len(user_outputs(emitter.timeline)) == 5


async def test_stop_during_stream_cannot_write_after_revision_rollback(
    chat, monkeypatch
):
    app, agent, emitter = chat
    session = emitter.session
    before = await app.chainlit_layer.get_thread("chat-1")
    started, restored = asyncio.Event(), asyncio.Event()
    cancelled_write = asyncio.Event()
    stop_task = None
    original_update_step = app.chainlit_layer.update_step

    async def observe_write(step):
        await original_update_step(step)
        if "Выполнение остановлено пользователем" in str(step.get("output", "")):
            cancelled_write.set()

    monkeypatch.setattr(app.chainlit_layer, "update_step", observe_write)

    async def stream(_chat_id, text, emit):
        await emit(TextDelta("Partial answer"))
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(agent, "run", stream)
    original_resume = emitter.resume_thread

    async def resume(thread):
        await original_resume(thread)
        if user_outputs(thread) == [f"request-{i}" for i in range(1, 6)]:
            restored.set()

    monkeypatch.setattr(emitter, "resume_thread", resume)
    original_cancel = ChainlitTurnView.cancel

    async def cancel(view):
        if asyncio.current_task() is stop_task:
            # A second writer outside the Turn can finish after rollback.
            # Force precisely that ordering, without timing-based sleeps.
            update = view.root.update

            async def delayed_update():
                await update()
                await asyncio.wait_for(restored.wait(), 3)

            monkeypatch.setattr(view.root, "update", delayed_update)
        await original_cancel(view)

    monkeypatch.setattr(ChainlitTurnView, "cancel", cancel)
    monkeypatch.setattr(WebsocketSession, "require", lambda _sid: session)
    monkeypatch.setattr(WebsocketSession, "get", lambda _sid: session)
    import local_agent_chat.chainlit_revision as adapter

    monkeypatch.setattr(adapter, "init_ws_context", lambda _s: context_var.get())
    monkeypatch.setattr(
        chainlit_socket, "init_ws_context", lambda _s: context_var.get()
    )
    await sio.handlers["/"]["edit_message"](
        "socket", {"message": {"id": "turn-3", "output": "stream"}}
    )
    await asyncio.wait_for(started.wait(), 3)
    stop_task = asyncio.create_task(sio.handlers["/"]["stop"]("socket"))
    await asyncio.wait_for(stop_task, 5)
    await asyncio.wait_for(session.current_task, 5)
    await asyncio.wait_for(cancelled_write.wait(), 3)
    after = await app.chainlit_layer.get_thread("chat-1")
    assert [(s["id"], s["output"]) for s in after["steps"]] == [
        (s["id"], s["output"]) for s in before["steps"]
    ]


async def test_edit_without_runtime_checkpoint_keeps_the_complete_history(chat):
    app, agent, emitter = chat
    await app.chainlit_layer.create_step(
        {
            "id": "unfinished",
            "threadId": "chat-1",
            "type": "user_message",
            "output": "unfinished",
            "createdAt": utc_now(),
        }
    )
    before = active_rows(app)
    await edit(app, "changed", "unfinished")
    assert active_rows(app) == before
    assert user_outputs(emitter.timeline)[-1] == "unfinished"
    with sqlite3.connect(app.settings.data_dir / "chainlit.sqlite3") as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM step_revisions").fetchone()[0] == 0
        )


async def test_busy_socket_edit_preserves_running_task_and_restores_ui(
    chat, monkeypatch
):
    app, agent, emitter = chat
    session = emitter.session
    monkeypatch.setattr(WebsocketSession, "require", lambda _sid: session)
    import local_agent_chat.chainlit_revision as adapter

    monkeypatch.setattr(adapter, "init_ws_context", lambda _session: context_var.get())
    running = asyncio.create_task(asyncio.Event().wait())
    session.current_task = running
    try:
        await sio.handlers["/"]["edit_message"](
            "socket", {"message": {"id": "turn-3", "output": "changed"}}
        )
        assert session.current_task is running and not running.done()
        assert len(active_rows(app)) == 5
        assert len(user_outputs(emitter.timeline)) == 5
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


def full_sqlite_state(app):
    result = {}
    for database, tables in {
        "chainlit.sqlite3": ["steps", "elements", "feedbacks"],
        "runtime-history.sqlite3": ["turns"],
    }.items():
        with sqlite3.connect(app.settings.data_dir / database) as connection:
            connection.row_factory = sqlite3.Row
            for table in tables:
                result[table] = sorted(
                    (dict(r) for r in connection.execute(f'SELECT * FROM "{table}"')),
                    key=lambda row: str(row.get("id", row.get("turn_id"))),
                )
    return result


@pytest.mark.parametrize("chat", [100], indirect=True)
async def test_hundred_turn_chat_repeated_edits_and_continuations(chat):
    app, agent, emitter = chat
    expected = [f"request-{i}" for i in range(1, 101)]
    for target in [100, 99, 75, 50, 26, 25, 2, 1]:
        before = active_rows(app)
        replacement = f"revision-{target}: 'quoted' \"double\" 🧪\nsecond line"
        await edit(app, replacement, f"turn-{target}")
        expected = expected[: target - 1] + [replacement]
        rows = active_rows(app)
        assert rows[: target - 1] == before[: target - 1]
        assert [r[1] for r in rows] == expected
        assert user_outputs(emitter.timeline) == agent.messages == expected
        assert (
            sum(s["type"] == "assistant_message" for s in emitter.timeline["steps"])
            == target
        )
        replacement += " repeated"
        await edit(app, replacement, f"turn-{target}")
        expected[-1] = replacement
        message = cl.Message(
            id=f"continued-{target}",
            type="user_message",
            content=f"continued-{target}",
            created_at=utc_now(),
        )
        await app.chainlit_layer.create_step(message.to_dict())
        await app.on_message(message)
        expected.append(message.content)
        assert agent.messages == expected
        assert [r[1] for r in active_rows(app)] == expected


@pytest.mark.parametrize("chat", [100], indirect=True)
@pytest.mark.parametrize(
    "failure", ["provider", "cancel", "write", "commit", "late_cancel"]
)
async def test_hundred_turn_revision_failures_restore_exact_state(
    chat, monkeypatch, failure
):
    app, agent, emitter = chat
    before = full_sqlite_state(app)
    if failure == "provider":
        await edit(app, "fail", "turn-2")
    elif failure == "cancel":
        task = asyncio.create_task(edit(app, "wait", "turn-2"))
        await asyncio.wait_for(agent.started.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif failure == "commit":

        async def fail_commit(_root):
            raise OSError("commit fault")

        monkeypatch.setattr(app.chainlit_layer, "_commit_revision", fail_commit)
        with pytest.raises(OSError, match="commit fault"):
            await edit(app, "changed", "turn-2")
    else:
        original_upsert = app.chainlit_layer._upsert_step
        reached = asyncio.Event()

        async def fail_write(session, step):
            if step.get("output") == "seen:request-1|changed":
                reached.set()
                if failure == "late_cancel":
                    await asyncio.Event().wait()
                raise OSError("write fault")
            await original_upsert(session, step)

        monkeypatch.setattr(app.chainlit_layer, "_upsert_step", fail_write)
        task = asyncio.create_task(edit(app, "changed", "turn-2"))
        await asyncio.wait_for(reached.wait(), 3)
        if failure == "late_cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(OSError, match="write fault"):
                await task
    assert full_sqlite_state(app) == before
    assert (
        agent.messages
        == user_outputs(emitter.timeline)
        == [f"request-{i}" for i in range(1, 101)]
    )
