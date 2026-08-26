import asyncio
import sqlite3
import tempfile
from pathlib import Path

import pytest
from chainlit.element import File
from chainlit.types import Feedback, Pagination, ThreadFilter
from chainlit.user import User

from local_agent_chat.chainlit_data import create_chainlit_data_layer
from local_agent_chat.chat_titles import (
    CHAT_TITLE_GENERATED,
    CHAT_TITLE_MANUAL,
    CHAT_TITLE_PENDING,
    CHAT_TITLE_STATE_KEY,
)
from local_agent_chat.local_storage import LocalStorageClient


async def _persist_file_element(
    layer,
    *,
    thread_id: str,
    step_id: str,
    element_id: str,
    name: str,
    content: bytes,
) -> dict:
    await layer.create_element.__wrapped__(
        layer,
        File(
            thread_id=thread_id,
            id=element_id,
            name=name,
            content=content,
            for_id=step_id,
            mime="text/plain",
        ),
    )
    stored = await layer.get_element(thread_id, element_id)
    assert stored is not None
    return stored


@pytest.mark.asyncio
async def test_existing_chainlit_database_is_migrated_for_current_step_fields(
    tmp_path: Path,
) -> None:
    database = tmp_path / "chainlit.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """CREATE TABLE steps (
                id TEXT PRIMARY KEY, name TEXT, type TEXT NOT NULL, "threadId" TEXT NOT NULL,
                "parentId" TEXT, streaming INTEGER DEFAULT 0, "waitForAnswer" INTEGER,
                "isError" INTEGER DEFAULT 0, metadata TEXT NOT NULL DEFAULT '{}', tags TEXT,
                input TEXT, output TEXT, "createdAt" TEXT, start TEXT, end TEXT,
                generation TEXT NOT NULL DEFAULT '{}', "showInput" TEXT, language TEXT,
                command TEXT, modes TEXT
            )"""
        )
        connection.execute(
            """CREATE TABLE step_revisions (
                "rootId" TEXT NOT NULL,
                id TEXT NOT NULL, name TEXT, type TEXT NOT NULL,
                "threadId" TEXT NOT NULL, "parentId" TEXT,
                streaming INTEGER, "waitForAnswer" INTEGER, "isError" INTEGER,
                metadata TEXT, tags TEXT, input TEXT, output TEXT,
                "createdAt" TEXT, start TEXT, end TEXT, generation TEXT,
                "showInput" TEXT, language TEXT, command TEXT, modes TEXT,
                "archivedAt" TEXT NOT NULL
            )"""
        )
        connection.execute(
            """INSERT INTO step_revisions (
                "rootId", id, type, "threadId", output, "createdAt", "archivedAt"
            ) VALUES (
                'legacy-root', 'legacy-root', 'user_message', 'legacy-chat',
                'legacy request', '2026-08-20T00:00:00Z', CURRENT_TIMESTAMP
            )"""
        )

    layer = create_chainlit_data_layer(database)
    user = await layer.create_user(User(identifier="local-user", metadata={}))
    assert user is not None
    await layer.update_thread("chat-1", user_id=user.id)
    await layer.create_step.__wrapped__(
        layer,
        {
            "id": "run-1",
            "name": "on_message",
            "type": "run",
            "threadId": "chat-1",
            "defaultOpen": False,
            "autoCollapse": False,
        },
    )

    stored = await layer.get_step("run-1")
    assert stored is not None
    await layer.close()

    with sqlite3.connect(database) as connection:
        step_revision_columns = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info(step_revisions)")
        }
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        archive_version = connection.execute(
            'SELECT "archiveVersion" FROM step_revisions WHERE id = ?',
            ("legacy-root",),
        ).fetchone()[0]

    assert step_revision_columns["archiveVersion"][4] == "1"
    assert archive_version == 1
    assert {"element_revisions", "feedback_revisions"} <= tables


@pytest.mark.asyncio
async def test_chainlit_history_is_available_after_data_layer_reopen() -> None:
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "chainlit.sqlite3"
        layer = create_chainlit_data_layer(database)
        user = await layer.create_user(User(identifier="local-user", metadata={}))
        assert user is not None
        await layer.update_thread(
            "chat-1",
            user_id=user.id,
            metadata={"model_profile": "local"},
        )
        await layer.update_thread("chat-1", name="First Chat")
        await layer.create_step.__wrapped__(
            layer,
            {
                "id": "message-1",
                "name": "user",
                "type": "user_message",
                "threadId": "chat-1",
                "output": "hello",
                "createdAt": "2026-08-21T00:00:00Z",
            },
        )
        await layer.close()

        reopened = create_chainlit_data_layer(database)
        page = await reopened.list_threads(
            Pagination(first=10), ThreadFilter(userId=user.id)
        )
        assert [(thread["id"], thread["name"]) for thread in page.data] == [
            ("chat-1", "First Chat")
        ]
        assert page.data[0]["steps"][0]["output"] == "hello"
        assert page.data[0]["metadata"] == {
            "model_profile": "local",
            CHAT_TITLE_STATE_KEY: CHAT_TITLE_MANUAL,
        }
        assert page.data[0]["steps"][0]["metadata"] == {}

        await reopened.update_step(
            {
                "id": "message-1",
                "name": "user",
                "type": "user_message",
                "threadId": "chat-1",
                "output": "edited",
                "createdAt": "2026-08-21T00:00:00Z",
            }
        )
        with pytest.raises(RuntimeError, match="abort revision"):
            async with reopened.revision("message-1"):
                raise RuntimeError("abort revision")
        restored = await reopened.get_step("message-1")
        assert restored is not None and restored["output"] == "hello"
        await reopened.close()


@pytest.mark.asyncio
async def test_generated_chat_title_replaces_and_outlives_chainlit_raw_name(
    tmp_path: Path,
) -> None:
    layer = create_chainlit_data_layer(tmp_path / "chainlit.sqlite3")
    user = await layer.create_user(User(identifier="local-user", metadata={}))
    assert user is not None

    await layer.update_thread(
        "chat-1",
        name="Первые слова полного пользовательского запроса",
        user_id=user.id,
    )
    initial = await layer.get_thread("chat-1")
    assert initial is not None
    assert initial["name"] == "Новый диалог"
    assert initial["metadata"] == {CHAT_TITLE_STATE_KEY: CHAT_TITLE_PENDING}

    await layer.update_thread("chat-1", metadata={"model_profile": "local"})
    assert await layer.complete_chat_title("chat-1", "Аудит проекта перед публикацией")
    await layer.update_thread(
        "chat-1",
        name="Поздняя запись сырого запроса",
        user_id=user.id,
    )
    generated = await layer.get_thread("chat-1")
    assert generated is not None
    assert generated["name"] == "Аудит проекта перед публикацией"
    assert generated["metadata"] == {
        CHAT_TITLE_STATE_KEY: CHAT_TITLE_GENERATED,
        "model_profile": "local",
    }

    await layer.update_thread("chat-1", name="Ручное название")
    renamed = await layer.get_thread("chat-1")
    assert renamed is not None and renamed["name"] == "Ручное название"
    assert renamed["metadata"][CHAT_TITLE_STATE_KEY] == CHAT_TITLE_MANUAL
    await layer.close()


@pytest.mark.asyncio
async def test_manual_rename_wins_over_pending_chat_title(tmp_path: Path) -> None:
    layer = create_chainlit_data_layer(tmp_path / "chainlit.sqlite3")
    user = await layer.create_user(User(identifier="local-user", metadata={}))
    assert user is not None
    await layer.update_thread("chat-1", name="raw request", user_id=user.id)

    await layer.update_thread("chat-1", name="Моё название")
    applied = await layer.complete_chat_title("chat-1", "Поздний заголовок модели")

    thread = await layer.get_thread("chat-1")
    assert applied is False
    assert thread is not None and thread["name"] == "Моё название"
    assert thread["metadata"][CHAT_TITLE_STATE_KEY] == CHAT_TITLE_MANUAL
    await layer.close()


@pytest.mark.asyncio
async def test_legacy_tool_logs_are_read_as_preformatted_text(tmp_path: Path) -> None:
    database = tmp_path / "chainlit.sqlite3"
    layer = create_chainlit_data_layer(database)
    user = await layer.create_user(User(identifier="local-user", metadata={}))
    assert user is not None
    await layer.update_thread("legacy-chat", user_id=user.id)
    await layer.create_step.__wrapped__(
        layer,
        {
            "id": "legacy-shell",
            "name": "Shell",
            "type": "tool",
            "threadId": "legacy-chat",
            "output": "\x1b[31mtotal 2\x1b[0m\n---\nfile.txt",
            "metadata": {},
            "createdAt": "2026-08-21T00:00:00Z",
        },
    )
    await layer.close()

    reopened = create_chainlit_data_layer(database)
    page = await reopened.list_threads(
        Pagination(first=10), ThreadFilter(userId=user.id)
    )
    tool = page.data[0]["steps"][0]

    assert tool["output"] == "```text\ntotal 2\n---\nfile.txt\n```"
    assert tool["metadata"]["tool_log_format"] == 1
    await reopened.close()


@pytest.mark.asyncio
async def test_revision_truncates_tool_steps_and_commits_replacement(
    tmp_path: Path,
) -> None:
    storage = LocalStorageClient(tmp_path / "blobs")
    layer = create_chainlit_data_layer(tmp_path / "chainlit.sqlite3", storage)
    user = await layer.create_user(User(identifier="local-user", metadata={}))
    assert user is not None
    await layer.update_thread("chat-1", user_id=user.id)
    common = {"threadId": "chat-1", "metadata": {}, "generation": {}}
    await layer.create_step.__wrapped__(
        layer,
        {
            **common,
            "id": "user-1",
            "name": "user",
            "type": "user_message",
            "output": "original",
            "createdAt": "2026-08-21T00:00:00.000Z",
        },
    )
    await layer.create_step.__wrapped__(
        layer,
        {
            **common,
            "id": "tool-1",
            "name": "Shell",
            "type": "tool",
            "parentId": "run-1",
            "output": "old tool output",
            "createdAt": "2026-08-21T00:00:01.000Z",
        },
    )
    await layer.create_step.__wrapped__(
        layer,
        {
            **common,
            "id": "answer-1",
            "name": "assistant",
            "type": "assistant_message",
            "output": "old answer",
            "createdAt": "2026-08-21T00:00:02.000Z",
        },
    )
    old_element = await _persist_file_element(
        layer,
        thread_id="chat-1",
        step_id="answer-1",
        element_id="element-1",
        name="old.txt",
        content=b"old blob",
    )
    await layer.upsert_feedback(
        Feedback(
            id="feedback-1",
            forId="answer-1",
            value=0,
            comment="old feedback",
        )
    )

    await layer.update_step(
        {
            **common,
            "id": "user-1",
            "name": "user",
            "type": "user_message",
            "output": "revised",
            "createdAt": "2026-08-21T00:00:00.000Z",
        }
    )
    async with layer.revision("user-1"):
        assert await layer.get_step("user-1") is not None
        assert await layer.get_step("tool-1") is None
        assert await layer.get_step("answer-1") is None

        await layer.create_step.__wrapped__(
            layer,
            {
                **common,
                "id": "answer-2",
                "name": "assistant",
                "type": "assistant_message",
                "output": "new answer",
                "createdAt": "2026-08-21T00:00:03.000Z",
            },
        )
        new_element = await _persist_file_element(
            layer,
            thread_id="chat-1",
            step_id="answer-2",
            element_id="element-2",
            name="new.txt",
            content=b"new blob",
        )
        await layer.upsert_feedback(
            Feedback(
                id="feedback-2",
                forId="answer-2",
                value=1,
                comment="new feedback",
            )
        )

    committed_answer = await layer.get_step("answer-2")
    assert committed_answer is not None
    assert committed_answer["output"] == "new answer"
    assert committed_answer["feedback"] == {
        "forId": "answer-2",
        "id": "feedback-2",
        "value": 1.0,
        "comment": "new feedback",
    }
    assert await layer.get_element("chat-1", "element-2") == new_element
    assert await layer.get_step("answer-1") is None
    assert await layer.get_element("chat-1", "element-1") is None
    assert not storage.path_for(old_element["objectKey"]).exists()
    assert storage.path_for(new_element["objectKey"]).read_bytes() == b"new blob"

    await layer.update_step(
        {
            **common,
            "id": "user-1",
            "name": "user",
            "type": "user_message",
            "output": "second revision",
            "createdAt": "2026-08-21T00:00:00.000Z",
        }
    )
    with pytest.raises(RuntimeError, match="abort second revision"):
        async with layer.revision("user-1"):
            raise RuntimeError("abort second revision")

    assert (await layer.get_step("user-1"))["output"] == "revised"
    assert await layer.get_step("answer-2") == committed_answer
    assert await layer.get_element("chat-1", "element-2") == new_element
    assert storage.path_for(new_element["objectKey"]).read_bytes() == b"new blob"
    await layer.close()


@pytest.mark.asyncio
async def test_cancellation_after_revision_commit_does_not_reopen_the_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = LocalStorageClient(tmp_path / "blobs")
    layer = create_chainlit_data_layer(tmp_path / "chainlit.sqlite3", storage)
    user = await layer.create_user(User(identifier="local-user", metadata={}))
    assert user is not None
    await layer.update_thread("chat-1", user_id=user.id)
    common = {"threadId": "chat-1", "metadata": {}, "generation": {}}
    await layer.create_step.__wrapped__(
        layer,
        {
            **common,
            "id": "user-1",
            "name": "user",
            "type": "user_message",
            "output": "original",
            "createdAt": "2026-08-21T00:00:00.000Z",
        },
    )
    await layer.create_step.__wrapped__(
        layer,
        {
            **common,
            "id": "answer-1",
            "name": "assistant",
            "type": "assistant_message",
            "output": "old answer",
            "createdAt": "2026-08-21T00:00:01.000Z",
        },
    )
    old_element = await _persist_file_element(
        layer,
        thread_id="chat-1",
        step_id="answer-1",
        element_id="element-1",
        name="old.txt",
        content=b"old blob",
    )
    await layer.update_step(
        {
            **common,
            "id": "user-1",
            "name": "user",
            "type": "user_message",
            "output": "revised",
            "createdAt": "2026-08-21T00:00:00.000Z",
        }
    )
    cleanup_started = asyncio.Event()

    async def interrupted_cleanup(_object_key: str) -> None:
        cleanup_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(storage, "delete_file", interrupted_cleanup)

    async def commit() -> None:
        async with layer.revision("user-1"):
            pass

    task = asyncio.create_task(commit())
    await cleanup_started.wait()
    task.cancel()
    await task

    assert (await layer.get_step("user-1"))["output"] == "revised"
    assert await layer.get_step("answer-1") is None
    assert storage.path_for(old_element["objectKey"]).exists()
    await layer.close()


@pytest.mark.asyncio
async def test_revision_restores_the_original_continuation_when_cancelled(
    tmp_path: Path,
) -> None:
    storage = LocalStorageClient(tmp_path / "blobs")
    layer = create_chainlit_data_layer(tmp_path / "chainlit.sqlite3", storage)
    user = await layer.create_user(User(identifier="local-user", metadata={}))
    assert user is not None
    await layer.update_thread("chat-1", user_id=user.id)
    common = {"threadId": "chat-1", "metadata": {}, "generation": {}}
    await layer.create_step.__wrapped__(
        layer,
        {
            **common,
            "id": "user-1",
            "name": "user",
            "type": "user_message",
            "output": "original",
            "createdAt": "2026-08-21T00:00:00.000Z",
        },
    )
    await layer.create_step.__wrapped__(
        layer,
        {
            **common,
            "id": "answer-1",
            "name": "assistant",
            "type": "assistant_message",
            "output": "old answer",
            "createdAt": "2026-08-21T00:00:01.000Z",
        },
    )
    original_element = await _persist_file_element(
        layer,
        thread_id="chat-1",
        step_id="answer-1",
        element_id="element-1",
        name="old.txt",
        content=b"old blob",
    )
    await layer.upsert_feedback(
        Feedback(
            id="feedback-1",
            forId="answer-1",
            value=1,
            comment="old feedback",
        )
    )
    original_answer = await layer.get_step("answer-1")
    assert original_answer is not None
    await layer.update_step(
        {
            **common,
            "id": "user-1",
            "name": "user",
            "type": "user_message",
            "output": "revised",
            "createdAt": "2026-08-21T00:00:00.000Z",
        }
    )

    entered = asyncio.Event()
    tentative_object_key = ""

    async def revise() -> None:
        nonlocal tentative_object_key
        async with layer.revision("user-1"):
            assert await layer.get_step("answer-1") is None
            await layer.create_step.__wrapped__(
                layer,
                {
                    **common,
                    "id": "answer-2",
                    "name": "assistant",
                    "type": "assistant_message",
                    "output": "tentative answer",
                    "createdAt": "2026-08-21T00:00:02.000Z",
                },
            )
            tentative_element = await _persist_file_element(
                layer,
                thread_id="chat-1",
                step_id="answer-2",
                element_id="element-2",
                name="new.txt",
                content=b"tentative blob",
            )
            tentative_object_key = tentative_element["objectKey"]
            await layer.upsert_feedback(
                Feedback(
                    id="feedback-2",
                    forId="answer-2",
                    value=0,
                    comment="tentative feedback",
                )
            )
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(revise())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (await layer.get_step("user-1"))["output"] == "original"
    assert await layer.get_step("answer-1") == original_answer
    assert await layer.get_element("chat-1", "element-1") == original_element
    assert await layer.get_step("answer-2") is None
    assert await layer.get_element("chat-1", "element-2") is None
    assert storage.path_for(original_element["objectKey"]).read_bytes() == b"old blob"
    assert not storage.path_for(tentative_object_key).exists()
    await layer.close()


@pytest.mark.asyncio
async def test_revision_exception_restores_the_first_complete_snapshot(
    tmp_path: Path,
) -> None:
    storage = LocalStorageClient(tmp_path / "blobs")
    layer = create_chainlit_data_layer(tmp_path / "chainlit.sqlite3", storage)
    user = await layer.create_user(User(identifier="local-user", metadata={}))
    assert user is not None
    await layer.update_thread("chat-1", user_id=user.id)
    common = {"threadId": "chat-1", "metadata": {}, "generation": {}}
    await layer.create_step.__wrapped__(
        layer,
        {
            **common,
            "id": "user-1",
            "name": "user",
            "type": "user_message",
            "output": "original",
            "createdAt": "2026-08-21T00:00:00.000Z",
        },
    )
    await layer.create_step.__wrapped__(
        layer,
        {
            **common,
            "id": "answer-1",
            "name": "assistant",
            "type": "assistant_message",
            "output": "old answer",
            "createdAt": "2026-08-21T00:00:01.000Z",
        },
    )
    original_element = await _persist_file_element(
        layer,
        thread_id="chat-1",
        step_id="answer-1",
        element_id="element-1",
        name="old.txt",
        content=b"old blob",
    )
    await layer.upsert_feedback(
        Feedback(
            id="feedback-1",
            forId="answer-1",
            value=1,
            comment="old feedback",
        )
    )
    original_answer = await layer.get_step("answer-1")
    assert original_answer is not None

    await layer.update_step(
        {
            **common,
            "id": "user-1",
            "name": "user",
            "type": "user_message",
            "output": "first revision",
            "createdAt": "2026-08-21T00:00:00.000Z",
        }
    )
    await layer.update_step(
        {
            **common,
            "id": "user-1",
            "name": "user",
            "type": "user_message",
            "output": "second revision",
            "createdAt": "2026-08-21T00:00:00.000Z",
        }
    )

    tentative_object_key = ""
    with pytest.raises(RuntimeError, match="provider failed"):
        async with layer.revision("user-1"):
            await layer.create_step.__wrapped__(
                layer,
                {
                    **common,
                    "id": "answer-2",
                    "name": "assistant",
                    "type": "assistant_message",
                    "output": "tentative answer",
                    "createdAt": "2026-08-21T00:00:02.000Z",
                },
            )
            tentative_element = await _persist_file_element(
                layer,
                thread_id="chat-1",
                step_id="answer-2",
                element_id="element-2",
                name="new.txt",
                content=b"tentative blob",
            )
            tentative_object_key = tentative_element["objectKey"]
            await layer.upsert_feedback(
                Feedback(
                    id="feedback-2",
                    forId="answer-2",
                    value=0,
                    comment="tentative feedback",
                )
            )
            raise RuntimeError("provider failed")

    assert (await layer.get_step("user-1"))["output"] == "original"
    assert await layer.get_step("answer-1") == original_answer
    assert await layer.get_element("chat-1", "element-1") == original_element
    assert await layer.get_step("answer-2") is None
    assert await layer.get_element("chat-1", "element-2") is None
    assert storage.path_for(original_element["objectKey"]).read_bytes() == b"old blob"
    assert not storage.path_for(tentative_object_key).exists()
    await layer.close()
