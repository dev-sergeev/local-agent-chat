from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data.storage_clients.base import BaseStorageClient
from chainlit.data.utils import queue_until_user_message

from .tool_logs import format_tool_log

STEP_COLUMNS = [
    "id",
    "name",
    "type",
    "threadId",
    "parentId",
    "streaming",
    "waitForAnswer",
    "isError",
    "metadata",
    "tags",
    "input",
    "output",
    "createdAt",
    "start",
    "end",
    "generation",
    "showInput",
    "language",
    "command",
    "modes",
    "defaultOpen",
    "autoCollapse",
]


class SQLiteChainlitDataLayer(SQLAlchemyDataLayer):
    chat_cleanup = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._step_locks: dict[str, asyncio.Lock] = {}

    def _step_lock(self, step_id: str) -> asyncio.Lock:
        return self._step_locks.setdefault(step_id, asyncio.Lock())

    @queue_until_user_message()
    async def create_step(self, step_dict):
        async with self._step_lock(step_dict["id"]):
            await SQLAlchemyDataLayer.create_step.__wrapped__(self, step_dict)

    async def get_all_user_threads(self, user_id=None, thread_id=None):
        threads = await super().get_all_user_threads(
            user_id=user_id, thread_id=thread_id
        )
        for thread in threads or []:
            for key in ("metadata", "tags"):
                if isinstance(thread.get(key), str):
                    thread[key] = json.loads(thread[key])
            for step in thread.get("steps", []):
                for key in ("metadata", "generation", "tags", "modes"):
                    if isinstance(step.get(key), str):
                        step[key] = json.loads(step[key])
                metadata = step.get("metadata") or {}
                if (
                    step.get("type") == "tool"
                    and not metadata.get("tool_log_format")
                    and step.get("output")
                ):
                    step["output"] = format_tool_log(str(step["output"]), limit=6000)
                    metadata["tool_log_format"] = 1
                    step["metadata"] = metadata
        return threads

    async def update_step(self, step_dict):
        async with self._step_lock(step_dict["id"]):
            existing = await self.get_step(step_dict["id"])
            if (
                existing
                and existing.get("type") == "user_message"
                and existing.get("output") != step_dict.get("output")
            ):
                await self.execute_sql(
                    query='DELETE FROM step_revisions WHERE "rootId" = :root_id',
                    parameters={"root_id": step_dict["id"]},
                )
                names = ", ".join(f'"{name}"' for name in STEP_COLUMNS)
                selected = ", ".join(f's."{name}"' for name in STEP_COLUMNS)
                await self.execute_sql(
                    query=f"""INSERT INTO step_revisions
                        ("rootId", {names}, "archivedAt")
                        SELECT :root_id, {selected}, CURRENT_TIMESTAMP FROM steps s
                        WHERE s."threadId" = :thread_id
                            AND s."createdAt" >= :created_at""",
                    parameters={
                        "root_id": step_dict["id"],
                        "thread_id": existing["threadId"],
                        "created_at": existing["createdAt"],
                    },
                )
            await SQLAlchemyDataLayer.create_step.__wrapped__(self, step_dict)

    async def wait_for_revision(self, root_id: str, timeout: float = 2.0) -> None:
        """Wait for Chainlit's background update task to stage the UI branch."""

        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            rows = await self.execute_sql(
                query='SELECT 1 FROM step_revisions WHERE "rootId" = :root_id LIMIT 1',
                parameters={"root_id": root_id},
            )
            if isinstance(rows, list) and rows:
                return
            await asyncio.sleep(0.01)
        raise RuntimeError("Timed out while staging the edited UI branch")

    async def truncate_revision(self, root_id: str) -> None:
        """Keep the edited user step and remove every superseded descendant."""

        root = await self.get_step(root_id)
        if root is None:
            raise KeyError(root_id)
        parameters = {
            "thread_id": root["threadId"],
            "created_at": root["createdAt"],
        }
        descendant_ids = """SELECT id FROM steps
            WHERE "threadId" = :thread_id AND "createdAt" > :created_at"""
        await self.execute_sql(
            query=f'DELETE FROM feedbacks WHERE "forId" IN ({descendant_ids})',
            parameters=parameters,
        )
        await self.execute_sql(
            query=f'DELETE FROM elements WHERE "forId" IN ({descendant_ids})',
            parameters=parameters,
        )
        await self.execute_sql(
            query='DELETE FROM steps WHERE "threadId" = :thread_id AND "createdAt" > :created_at',
            parameters=parameters,
        )

    async def commit_revision(self, root_id: str) -> None:
        await self.execute_sql(
            query='DELETE FROM step_revisions WHERE "rootId" = :root_id',
            parameters={"root_id": root_id},
        )

    async def restore_revision(self, root_id: str) -> None:
        rows = await self.execute_sql(
            query='SELECT * FROM step_revisions WHERE "rootId" = :root_id ORDER BY "createdAt"',
            parameters={"root_id": root_id},
        )
        if not isinstance(rows, list) or not rows:
            return
        first = rows[0]
        await self.execute_sql(
            query='DELETE FROM steps WHERE "threadId" = :thread_id AND "createdAt" >= :created_at',
            parameters={
                "thread_id": first["threadId"],
                "created_at": first["createdAt"],
            },
        )
        for row in rows:
            names = ",".join(f'"{name}"' for name in STEP_COLUMNS)
            values = ",".join(f":{name}" for name in STEP_COLUMNS)
            await self.execute_sql(
                query=f"INSERT OR REPLACE INTO steps ({names}) VALUES ({values})",
                parameters={name: row.get(name) for name in STEP_COLUMNS},
            )
        await self.commit_revision(root_id)

    async def delete_thread(self, thread_id: str):
        await super().delete_thread(thread_id)
        if self.chat_cleanup is not None:
            await self.chat_cleanup(thread_id)


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS users (
 id TEXT PRIMARY KEY, identifier TEXT UNIQUE NOT NULL, "createdAt" TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS threads (
 id TEXT PRIMARY KEY, "createdAt" TEXT, name TEXT, "userId" TEXT, "userIdentifier" TEXT,
 tags TEXT, metadata TEXT NOT NULL DEFAULT '{}', FOREIGN KEY("userId") REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS steps (
 id TEXT PRIMARY KEY, name TEXT, type TEXT NOT NULL, "threadId" TEXT NOT NULL, "parentId" TEXT,
 streaming INTEGER DEFAULT 0, "waitForAnswer" INTEGER, "isError" INTEGER DEFAULT 0,
 metadata TEXT NOT NULL DEFAULT '{}', tags TEXT, input TEXT, output TEXT, "createdAt" TEXT,
 start TEXT, end TEXT, generation TEXT NOT NULL DEFAULT '{}', "showInput" TEXT, language TEXT,
 command TEXT, modes TEXT, "defaultOpen" INTEGER DEFAULT 0,
 "autoCollapse" INTEGER DEFAULT 0,
 FOREIGN KEY("threadId") REFERENCES threads(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_steps_thread ON steps("threadId", "createdAt");
CREATE TABLE IF NOT EXISTS feedbacks (
 id TEXT PRIMARY KEY, "forId" TEXT NOT NULL, value REAL, comment TEXT
);
CREATE TABLE IF NOT EXISTS elements (
 id TEXT PRIMARY KEY, "threadId" TEXT, type TEXT NOT NULL, "chainlitKey" TEXT, url TEXT,
 "objectKey" TEXT, name TEXT NOT NULL, display TEXT, size INTEGER, language TEXT, page INTEGER,
 "autoPlay" INTEGER, "playerConfig" TEXT, "forId" TEXT, mime TEXT, props TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS step_revisions (
 "rootId" TEXT NOT NULL,
 id TEXT NOT NULL, name TEXT, type TEXT NOT NULL, "threadId" TEXT NOT NULL, "parentId" TEXT,
 streaming INTEGER, "waitForAnswer" INTEGER, "isError" INTEGER, metadata TEXT, tags TEXT,
 input TEXT, output TEXT, "createdAt" TEXT, start TEXT, end TEXT, generation TEXT, "showInput" TEXT,
 language TEXT, command TEXT, modes TEXT, "defaultOpen" INTEGER DEFAULT 0,
 "autoCollapse" INTEGER DEFAULT 0, "archivedAt" TEXT NOT NULL
);
"""


def _migrate_schema(connection: sqlite3.Connection) -> None:
    for table in ("steps", "step_revisions"):
        existing = {
            row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')
        }
        for column in ("defaultOpen", "autoCollapse"):
            if column not in existing:
                connection.execute(
                    f'ALTER TABLE "{table}" ADD COLUMN "{column}" INTEGER DEFAULT 0'
                )


def create_chainlit_data_layer(
    path: Path, storage: BaseStorageClient | None = None
) -> SQLAlchemyDataLayer:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA)
        _migrate_schema(connection)
    return SQLiteChainlitDataLayer(
        conninfo=f"sqlite+aiosqlite:///{path}", storage_provider=storage
    )
