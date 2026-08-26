from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data.storage_clients.base import BaseStorageClient
from chainlit.data.utils import queue_until_user_message
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .chat_titles import (
    CHAT_TITLE_FALLBACK,
    CHAT_TITLE_GENERATED,
    CHAT_TITLE_MANUAL,
    CHAT_TITLE_PENDING,
    CHAT_TITLE_STATE_KEY,
    DEFAULT_CHAT_TITLE,
    chat_title_source,
)
from .tool_logs import format_tool_log

logger = logging.getLogger(__name__)

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

ELEMENT_COLUMNS = [
    "id",
    "threadId",
    "type",
    "chainlitKey",
    "url",
    "objectKey",
    "name",
    "display",
    "size",
    "language",
    "page",
    "autoPlay",
    "playerConfig",
    "forId",
    "mime",
    "props",
]

FEEDBACK_COLUMNS = ["id", "forId", "value", "comment"]
REVISION_ARCHIVE_VERSION = 2


class SQLiteChainlitDataLayer(SQLAlchemyDataLayer):
    chat_cleanup = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._step_locks: dict[str, asyncio.Lock] = {}
        self._thread_locks: dict[str, asyncio.Lock] = {}
        self._initial_name_events: dict[str, asyncio.Event] = {}

    def _step_lock(self, step_id: str) -> asyncio.Lock:
        return self._step_locks.setdefault(step_id, asyncio.Lock())

    def _thread_lock(self, thread_id: str) -> asyncio.Lock:
        return self._thread_locks.setdefault(thread_id, asyncio.Lock())

    async def wait_for_initial_name(self, thread_id: str, timeout: float = 1.0) -> None:
        """Let Chainlit finish its first-interaction event before replacing its title."""

        event = self._initial_name_events.setdefault(thread_id, asyncio.Event())
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except TimeoutError:
            return
        await asyncio.sleep(0)

    async def _chat_title_record(self, thread_id: str) -> tuple[str | None, dict]:
        rows = await self.execute_sql(
            query='SELECT name, metadata FROM threads WHERE "id" = :thread_id',
            parameters={"thread_id": thread_id},
        )
        if not isinstance(rows, list) or not rows:
            return None, {}
        metadata = rows[0].get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        return rows[0].get("name"), metadata

    async def chat_title_state(self, thread_id: str) -> str | None:
        _name, metadata = await self._chat_title_record(thread_id)
        state = metadata.get(CHAT_TITLE_STATE_KEY)
        return str(state) if state else None

    async def begin_chat_title(self, thread_id: str) -> bool:
        """Claim an unnamed Chat for automatic title generation."""

        async with self._thread_lock(thread_id):
            _name, metadata = await self._chat_title_record(thread_id)
            state = metadata.get(CHAT_TITLE_STATE_KEY)
            if state == CHAT_TITLE_PENDING:
                return True
            if state == CHAT_TITLE_FALLBACK:
                await super().update_thread(
                    thread_id,
                    metadata={**metadata, CHAT_TITLE_STATE_KEY: CHAT_TITLE_PENDING},
                )
                return True
            if state is not None:
                return False
            await super().update_thread(
                thread_id,
                name=DEFAULT_CHAT_TITLE,
                metadata={CHAT_TITLE_STATE_KEY: CHAT_TITLE_PENDING},
            )
            return True

    async def complete_chat_title(
        self, thread_id: str, title: str, *, fallback: bool = False
    ) -> bool:
        """Apply an automatic title only while the placeholder is untouched."""

        state = CHAT_TITLE_FALLBACK if fallback else CHAT_TITLE_GENERATED
        async with self._thread_lock(thread_id):
            _name, metadata = await self._chat_title_record(thread_id)
            if metadata.get(CHAT_TITLE_STATE_KEY) != CHAT_TITLE_PENDING:
                return False
            await super().update_thread(
                thread_id,
                name=title,
                metadata={CHAT_TITLE_STATE_KEY: state},
            )
            return True

    async def first_user_request(self, thread_id: str) -> str | None:
        rows = await self.execute_sql(
            query="""SELECT id, output FROM steps
                WHERE "threadId" = :thread_id AND type = 'user_message'
                ORDER BY "createdAt" ASC LIMIT 1""",
            parameters={"thread_id": thread_id},
        )
        if not isinstance(rows, list) or not rows:
            return None
        output = rows[0].get("output")
        elements = await self.execute_sql(
            query="""SELECT name FROM elements
                WHERE "forId" = :step_id AND name IS NOT NULL
                ORDER BY id LIMIT 5""",
            parameters={"step_id": rows[0]["id"]},
        )
        filenames = (
            [str(element["name"]) for element in elements if element.get("name")]
            if isinstance(elements, list)
            else []
        )
        source = chat_title_source(str(output or ""), filenames)
        return source or None

    async def has_user_request(self, thread_id: str) -> bool:
        rows = await self.execute_sql(
            query="""SELECT 1 AS present FROM steps
                WHERE "threadId" = :thread_id AND type = 'user_message'
                LIMIT 1""",
            parameters={"thread_id": thread_id},
        )
        return isinstance(rows, list) and bool(rows)

    async def update_thread(
        self,
        thread_id,
        name=None,
        user_id=None,
        metadata=None,
        tags=None,
    ):
        is_initial_name = name is not None and user_id is not None
        async with self._thread_lock(thread_id):
            _stored_name, stored_metadata = await self._chat_title_record(thread_id)
            title_state = stored_metadata.get(CHAT_TITLE_STATE_KEY)
            if is_initial_name:
                # Chainlit supplies the raw first request as name. Store a neutral
                # placeholder and let the bounded title task replace it later.
                if title_state in {
                    CHAT_TITLE_GENERATED,
                    CHAT_TITLE_FALLBACK,
                    CHAT_TITLE_MANUAL,
                }:
                    name = None
                else:
                    name = DEFAULT_CHAT_TITLE
                    metadata = {
                        **(metadata or {}),
                        CHAT_TITLE_STATE_KEY: CHAT_TITLE_PENDING,
                    }
            elif name is not None:
                # Public name-only updates come from Chainlit's manual rename API.
                metadata = {
                    **(metadata or {}),
                    CHAT_TITLE_STATE_KEY: CHAT_TITLE_MANUAL,
                }
            await super().update_thread(
                thread_id,
                name=name,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
            )
            if is_initial_name:
                self._initial_name_events.setdefault(thread_id, asyncio.Event()).set()

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
            await self.update_thread(step_dict["threadId"])
            existing = await self.get_step(step_dict["id"])
            async with self.async_session() as session:
                async with session.begin():
                    if (
                        existing
                        and existing.get("type") == "user_message"
                        and existing.get("output") != step_dict.get("output")
                    ):
                        await self._stage_revision(session, step_dict["id"], existing)
                    await self._upsert_step(session, step_dict)

    @staticmethod
    async def _upsert_step(session: AsyncSession, step_dict: dict) -> None:
        record = dict(step_dict)
        record["showInput"] = (
            str(record.get("showInput", "")).lower() if "showInput" in record else None
        )
        parameters = {
            key: value
            for key, value in record.items()
            if value is not None and not (isinstance(value, dict) and not value)
        }
        parameters["metadata"] = json.dumps(record.get("metadata", {}))
        parameters["generation"] = json.dumps(record.get("generation", {}))
        columns = ", ".join(f'"{key}"' for key in parameters)
        values = ", ".join(f":{key}" for key in parameters)
        updates = ", ".join(f'"{key}" = :{key}' for key in parameters if key != "id")
        await session.execute(
            text(
                f"""INSERT INTO steps ({columns})
                    VALUES ({values})
                    ON CONFLICT (id) DO UPDATE SET {updates}"""
            ),
            parameters,
        )

    @staticmethod
    async def _stage_revision(
        session: AsyncSession, root_id: str, existing: dict
    ) -> None:
        staged = await session.execute(
            text('SELECT 1 FROM step_revisions WHERE "rootId" = :root_id LIMIT 1'),
            {"root_id": root_id},
        )
        if staged.first() is not None:
            return

        parameters = {
            "root_id": root_id,
            "thread_id": existing["threadId"],
            "created_at": existing["createdAt"],
            "archive_version": REVISION_ARCHIVE_VERSION,
        }
        step_names = ", ".join(f'"{name}"' for name in STEP_COLUMNS)
        selected_steps = ", ".join(f's."{name}"' for name in STEP_COLUMNS)
        await session.execute(
            text(
                f"""INSERT INTO step_revisions
                    ("rootId", {step_names}, "archivedAt", "archiveVersion")
                    SELECT :root_id, {selected_steps}, CURRENT_TIMESTAMP,
                           :archive_version
                    FROM steps s
                    WHERE s."threadId" = :thread_id
                      AND s."createdAt" >= :created_at"""
            ),
            parameters,
        )

        element_names = ", ".join(f'"{name}"' for name in ELEMENT_COLUMNS)
        selected_elements = ", ".join(f'e."{name}"' for name in ELEMENT_COLUMNS)
        await session.execute(
            text(
                f"""INSERT INTO element_revisions
                    ("rootId", {element_names}, "archivedAt")
                    SELECT :root_id, {selected_elements}, CURRENT_TIMESTAMP
                    FROM elements e
                    WHERE e."forId" IN (
                        SELECT id FROM step_revisions
                        WHERE "rootId" = :root_id
                    )"""
            ),
            parameters,
        )

        feedback_names = ", ".join(f'"{name}"' for name in FEEDBACK_COLUMNS)
        selected_feedback = ", ".join(f'f."{name}"' for name in FEEDBACK_COLUMNS)
        await session.execute(
            text(
                f"""INSERT INTO feedback_revisions
                    ("rootId", {feedback_names}, "archivedAt")
                    SELECT :root_id, {selected_feedback}, CURRENT_TIMESTAMP
                    FROM feedbacks f
                    WHERE f."forId" IN (
                        SELECT id FROM step_revisions
                        WHERE "rootId" = :root_id
                    )"""
            ),
            parameters,
        )

    @asynccontextmanager
    async def revision(self, root_id: str) -> AsyncIterator[None]:
        """Replace one persisted UI continuation or restore it on interruption."""

        await self._wait_for_revision(root_id)
        try:
            await self._truncate_revision(root_id)
            yield
        except BaseException:
            await asyncio.shield(self._restore_revision(root_id))
            raise
        try:
            await self._commit_revision(root_id)
        except BaseException:
            await asyncio.shield(self._restore_revision(root_id))
            raise

    async def _wait_for_revision(self, root_id: str, timeout: float = 2.0) -> None:
        """Wait for Chainlit to stage the superseded UI continuation."""

        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            async with self.async_session() as session:
                staged = await session.execute(
                    text(
                        'SELECT 1 FROM step_revisions WHERE "rootId" = :root_id LIMIT 1'
                    ),
                    {"root_id": root_id},
                )
            if staged.first() is not None:
                return
            await asyncio.sleep(0.01)
        raise RuntimeError("Timed out while staging the edited UI continuation")

    @staticmethod
    async def _revision_root(session: AsyncSession, root_id: str):
        result = await session.execute(
            text(
                """SELECT "threadId", "createdAt", "archiveVersion"
                   FROM step_revisions
                   WHERE "rootId" = :root_id AND id = :root_id"""
            ),
            {"root_id": root_id},
        )
        root = result.mappings().first()
        if root is None:
            raise KeyError(root_id)
        return root

    async def _truncate_revision(self, root_id: str) -> None:
        """Keep the edited user step and remove every superseded descendant."""

        async with self.async_session() as session:
            async with session.begin():
                root = await self._revision_root(session, root_id)
                parameters = {
                    "thread_id": root["threadId"],
                    "created_at": root["createdAt"],
                }
                descendant_ids = """SELECT id FROM steps
                    WHERE "threadId" = :thread_id
                      AND "createdAt" > :created_at"""
                await session.execute(
                    text(f'DELETE FROM feedbacks WHERE "forId" IN ({descendant_ids})'),
                    parameters,
                )
                await session.execute(
                    text(f'DELETE FROM elements WHERE "forId" IN ({descendant_ids})'),
                    parameters,
                )
                await session.execute(
                    text(
                        'DELETE FROM steps WHERE "threadId" = :thread_id '
                        'AND "createdAt" > :created_at'
                    ),
                    parameters,
                )

    async def _commit_revision(self, root_id: str) -> None:
        object_keys: set[str] = set()
        async with self.async_session() as session:
            async with session.begin():
                stored_keys = await session.execute(
                    text(
                        'SELECT DISTINCT "objectKey" FROM element_revisions '
                        'WHERE "rootId" = :root_id AND "objectKey" IS NOT NULL'
                    ),
                    {"root_id": root_id},
                )
                object_keys = {str(row[0]) for row in stored_keys}
                await session.execute(
                    text('DELETE FROM feedback_revisions WHERE "rootId" = :root_id'),
                    {"root_id": root_id},
                )
                await session.execute(
                    text('DELETE FROM element_revisions WHERE "rootId" = :root_id'),
                    {"root_id": root_id},
                )
                await session.execute(
                    text('DELETE FROM step_revisions WHERE "rootId" = :root_id'),
                    {"root_id": root_id},
                )
        await self._delete_unreferenced_blobs(object_keys)

    async def _restore_revision(self, root_id: str) -> None:
        object_keys: set[str] = set()
        async with self.async_session() as session:
            async with session.begin():
                stored_steps = await session.execute(
                    text(
                        """SELECT * FROM step_revisions
                           WHERE "rootId" = :root_id
                           ORDER BY "createdAt", id"""
                    ),
                    {"root_id": root_id},
                )
                step_rows = list(stored_steps.mappings())
                if not step_rows:
                    return
                root = next(
                    (row for row in step_rows if row["id"] == root_id), step_rows[0]
                )
                version = int(root.get("archiveVersion") or 1)
                parameters = {
                    "thread_id": root["threadId"],
                    "created_at": root["createdAt"],
                    "root_id": root_id,
                }
                continuation_ids = """SELECT id FROM steps
                    WHERE "threadId" = :thread_id
                      AND "createdAt" >= :created_at"""

                element_rows = []
                feedback_rows = []
                if version >= REVISION_ARCHIVE_VERSION:
                    current_keys = await session.execute(
                        text(
                            f"""SELECT DISTINCT "objectKey" FROM elements
                                WHERE "forId" IN ({continuation_ids})
                                  AND "objectKey" IS NOT NULL"""
                        ),
                        parameters,
                    )
                    object_keys = {str(row[0]) for row in current_keys}
                    stored_elements = await session.execute(
                        text(
                            """SELECT * FROM element_revisions
                               WHERE "rootId" = :root_id ORDER BY id"""
                        ),
                        parameters,
                    )
                    element_rows = list(stored_elements.mappings())
                    stored_feedback = await session.execute(
                        text(
                            """SELECT * FROM feedback_revisions
                               WHERE "rootId" = :root_id ORDER BY id"""
                        ),
                        parameters,
                    )
                    feedback_rows = list(stored_feedback.mappings())
                    await session.execute(
                        text(
                            f"DELETE FROM feedbacks "
                            f'WHERE "forId" IN ({continuation_ids})'
                        ),
                        parameters,
                    )
                    await session.execute(
                        text(
                            f"DELETE FROM elements "
                            f'WHERE "forId" IN ({continuation_ids})'
                        ),
                        parameters,
                    )

                await session.execute(
                    text(
                        'DELETE FROM steps WHERE "threadId" = :thread_id '
                        'AND "createdAt" >= :created_at'
                    ),
                    parameters,
                )
                await self._insert_archived_rows(
                    session,
                    "steps",
                    STEP_COLUMNS,
                    step_rows,
                    replace=version < REVISION_ARCHIVE_VERSION,
                )
                if version >= REVISION_ARCHIVE_VERSION:
                    await self._insert_archived_rows(
                        session, "elements", ELEMENT_COLUMNS, element_rows
                    )
                    await self._insert_archived_rows(
                        session, "feedbacks", FEEDBACK_COLUMNS, feedback_rows
                    )
                    await session.execute(
                        text(
                            'DELETE FROM feedback_revisions WHERE "rootId" = :root_id'
                        ),
                        parameters,
                    )
                    await session.execute(
                        text('DELETE FROM element_revisions WHERE "rootId" = :root_id'),
                        parameters,
                    )
                await session.execute(
                    text('DELETE FROM step_revisions WHERE "rootId" = :root_id'),
                    parameters,
                )
        await self._delete_unreferenced_blobs(object_keys)

    @staticmethod
    async def _insert_archived_rows(
        session: AsyncSession,
        table: str,
        columns: list[str],
        rows: list,
        *,
        replace: bool = False,
    ) -> None:
        if not rows:
            return
        names = ", ".join(f'"{name}"' for name in columns)
        values = ", ".join(f":{name}" for name in columns)
        operation = "INSERT OR REPLACE" if replace else "INSERT"
        await session.execute(
            text(f"{operation} INTO {table} ({names}) VALUES ({values})"),
            [{name: row.get(name) for name in columns} for row in rows],
        )

    async def _delete_unreferenced_blobs(self, object_keys: set[str]) -> None:
        if self.storage_provider is None:
            return
        for object_key in sorted(object_keys):
            try:
                async with self.async_session() as session:
                    referenced = await session.execute(
                        text(
                            'SELECT 1 FROM elements WHERE "objectKey" = :object_key '
                            "LIMIT 1"
                        ),
                        {"object_key": object_key},
                    )
                if referenced.first() is None:
                    await self.storage_provider.delete_file(object_key)
            except asyncio.CancelledError:
                # The database decision is already committed. Keep the Turn
                # authoritative even if best-effort blob cleanup is interrupted.
                logger.warning(
                    "Revision blob cleanup was interrupted for %s", object_key
                )
                return
            except Exception:  # noqa: BLE001 - blob cleanup is post-decision
                logger.warning(
                    "Failed to delete an unreferenced Revision blob %s",
                    object_key,
                    exc_info=True,
                )

    async def delete_thread(self, thread_id: str):
        if self.chat_cleanup is not None:
            await self.chat_cleanup(thread_id)
        await super().delete_thread(thread_id)
        self._thread_locks.pop(thread_id, None)
        self._initial_name_events.pop(thread_id, None)


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
 "autoCollapse" INTEGER DEFAULT 0, "archivedAt" TEXT NOT NULL,
 "archiveVersion" INTEGER NOT NULL DEFAULT 2
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_step_revisions_root_id
ON step_revisions("rootId", id);
CREATE TABLE IF NOT EXISTS element_revisions (
 "rootId" TEXT NOT NULL,
 id TEXT NOT NULL, "threadId" TEXT, type TEXT NOT NULL, "chainlitKey" TEXT, url TEXT,
 "objectKey" TEXT, name TEXT NOT NULL, display TEXT, size INTEGER, language TEXT, page INTEGER,
 "autoPlay" INTEGER, "playerConfig" TEXT, "forId" TEXT, mime TEXT,
 props TEXT NOT NULL DEFAULT '{}', "archivedAt" TEXT NOT NULL,
 PRIMARY KEY("rootId", id)
);
CREATE TABLE IF NOT EXISTS feedback_revisions (
 "rootId" TEXT NOT NULL,
 id TEXT NOT NULL, "forId" TEXT NOT NULL, value REAL, comment TEXT,
 "archivedAt" TEXT NOT NULL,
 PRIMARY KEY("rootId", id)
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
    revision_columns = {
        row[1] for row in connection.execute('PRAGMA table_info("step_revisions")')
    }
    if "archiveVersion" not in revision_columns:
        connection.execute(
            'ALTER TABLE "step_revisions" '
            'ADD COLUMN "archiveVersion" INTEGER NOT NULL DEFAULT 1'
        )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_step_revisions_root_id "
        'ON step_revisions("rootId", id)'
    )


def create_chainlit_data_layer(
    path: Path, storage: BaseStorageClient | None = None
) -> SQLiteChainlitDataLayer:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA)
        _migrate_schema(connection)
    return SQLiteChainlitDataLayer(
        conninfo=f"sqlite+aiosqlite:///{path}", storage_provider=storage
    )
