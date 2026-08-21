import sqlite3
import tempfile
from pathlib import Path

import pytest
from chainlit.types import Pagination, ThreadFilter
from chainlit.user import User

from local_agent_chat.chainlit_data import create_chainlit_data_layer


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


@pytest.mark.asyncio
async def test_chainlit_history_is_available_after_data_layer_reopen() -> None:
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "chainlit.sqlite3"
        layer = create_chainlit_data_layer(database)
        user = await layer.create_user(User(identifier="local-user", metadata={}))
        assert user is not None
        await layer.update_thread(
            "chat-1",
            name="First Chat",
            user_id=user.id,
            metadata={"model_profile": "local"},
        )
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
        assert page.data[0]["metadata"] == {"model_profile": "local"}
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
        await reopened.restore_revision("message-1")
        restored = await reopened.get_step("message-1")
        assert restored is not None and restored["output"] == "hello"
        await reopened.close()


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
    layer = create_chainlit_data_layer(tmp_path / "chainlit.sqlite3")
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
    await layer.wait_for_revision("user-1")
    await layer.truncate_revision("user-1")

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
    await layer.commit_revision("user-1")
    staged = await layer.execute_sql(
        query='SELECT * FROM step_revisions WHERE "rootId" = :root_id',
        parameters={"root_id": "user-1"},
    )

    assert staged == []
    assert (await layer.get_step("answer-2"))["output"] == "new answer"
    await layer.close()
