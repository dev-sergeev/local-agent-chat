from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .agent_modes import AgentMode


@dataclass(frozen=True, slots=True)
class ChatBinding:
    profile_id: str
    mode: AgentMode
    mode_locked: bool
    memory_thread_id: str


class ChatBindings:
    """Persist the immutable Model Profile and Agent Mode of each Chat."""

    def __init__(self, database: Path, available_profile_ids: Iterable[str]) -> None:
        profiles = tuple(dict.fromkeys(available_profile_ids))
        if not profiles:
            raise ValueError("At least one Model Profile must be available")
        if any(not profile_id for profile_id in profiles):
            raise ValueError("Model Profile identifiers must not be empty")

        self._database = database
        self._available_profiles = frozenset(profiles)
        self._deleting: set[str] = set()
        self._database.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database)

    def _migrate(self) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active_branches = connection.execute(
                """SELECT 1 FROM sqlite_master
                   WHERE type = 'table' AND name = 'active_branches'"""
            ).fetchone()
            if active_branches is not None:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(active_branches)")
                }
                legacy_registry = "agent_mode" not in columns
                if legacy_registry:
                    connection.execute(
                        "ALTER TABLE active_branches ADD COLUMN "
                        "agent_mode TEXT NOT NULL DEFAULT 'read_only'"
                    )
                if "mode_locked" not in columns:
                    connection.execute(
                        "ALTER TABLE active_branches ADD COLUMN "
                        "mode_locked INTEGER NOT NULL DEFAULT 0"
                    )
                if legacy_registry:
                    connection.execute(
                        "UPDATE active_branches "
                        "SET agent_mode = 'extended', mode_locked = 1"
                    )
                connection.execute(
                    """UPDATE active_branches SET agent_mode = 'read_only'
                       WHERE agent_mode IS NULL
                          OR agent_mode NOT IN ('read_only', 'extended')"""
                )

            checkpoint_ns_bindings = connection.execute(
                """SELECT 1 FROM sqlite_master
                   WHERE type = 'table' AND name = 'chat_bindings'"""
            ).fetchone()
            if checkpoint_ns_bindings is not None:
                binding_columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(chat_bindings)")
                }
                if "memory_thread_id" not in binding_columns:
                    connection.execute(
                        """UPDATE chat_bindings SET agent_mode = 'read_only'
                           WHERE agent_mode IS NULL
                              OR agent_mode NOT IN ('read_only', 'extended')"""
                    )
                    connection.execute(
                        "ALTER TABLE chat_bindings "
                        "RENAME TO checkpoint_ns_chat_bindings"
                    )

            connection.execute(
                """CREATE TABLE IF NOT EXISTS chat_bindings (
                       chat_id TEXT PRIMARY KEY,
                       profile_id TEXT NOT NULL,
                       memory_thread_id TEXT NOT NULL,
                       agent_mode TEXT NOT NULL DEFAULT 'read_only'
                           CHECK(agent_mode IN ('read_only', 'extended')),
                       mode_locked INTEGER NOT NULL DEFAULT 0
                           CHECK(mode_locked IN (0, 1))
                   )"""
            )
            checkpoint_ns_bindings = connection.execute(
                """SELECT 1 FROM sqlite_master
                   WHERE type = 'table'
                     AND name = 'checkpoint_ns_chat_bindings'"""
            ).fetchone()
            if checkpoint_ns_bindings is not None:
                connection.execute(
                    """INSERT OR IGNORE INTO chat_bindings(
                           chat_id, profile_id, memory_thread_id,
                           agent_mode, mode_locked
                       )
                       SELECT chat_id, profile_id, chat_id,
                              agent_mode, mode_locked
                       FROM checkpoint_ns_chat_bindings"""
                )
                connection.execute("DROP TABLE checkpoint_ns_chat_bindings")
            if active_branches is not None:
                connection.execute(
                    """INSERT OR IGNORE INTO chat_bindings(
                           chat_id, profile_id, memory_thread_id,
                           agent_mode, mode_locked
                       )
                       SELECT chat_id, profile_id, chat_id,
                              agent_mode, mode_locked
                       FROM active_branches"""
                )
                connection.execute("DROP TABLE active_branches")
            connection.execute(
                """UPDATE chat_bindings SET agent_mode = 'read_only'
                   WHERE agent_mode IS NULL
                      OR agent_mode NOT IN ('read_only', 'extended')"""
            )
            connection.execute(
                """UPDATE chat_bindings SET memory_thread_id = chat_id
                   WHERE memory_thread_id IS NULL OR memory_thread_id = ''"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS chat_memory_threads (
                       chat_id TEXT NOT NULL,
                       thread_id TEXT NOT NULL,
                       PRIMARY KEY(chat_id, thread_id)
                   )"""
            )
            connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS
                       idx_chat_memory_threads_thread_id
                   ON chat_memory_threads(thread_id)"""
            )
            connection.execute(
                """INSERT OR IGNORE INTO chat_memory_threads(chat_id, thread_id)
                   SELECT chat_id, memory_thread_id FROM chat_bindings"""
            )

    @staticmethod
    def _binding(row: tuple[object, ...]) -> ChatBinding:
        return ChatBinding(
            profile_id=str(row[0]),
            mode=AgentMode(str(row[1])),
            mode_locked=bool(row[2]),
            memory_thread_id=str(row[3]),
        )

    @staticmethod
    def _get_row(
        connection: sqlite3.Connection, chat_id: str
    ) -> tuple[object, ...] | None:
        return connection.execute(
            """SELECT profile_id, agent_mode, mode_locked, memory_thread_id
               FROM chat_bindings WHERE chat_id = ?""",
            (chat_id,),
        ).fetchone()

    def get(self, chat_id: str) -> ChatBinding | None:
        with self._connect() as connection:
            row = self._get_row(connection, chat_id)
        return self._binding(row) if row is not None else None

    def is_deleting(self, chat_id: str) -> bool:
        return chat_id in self._deleting

    def mark_deleting(self, chat_id: str) -> None:
        self._deleting.add(chat_id)

    def _ensure_active(self, chat_id: str) -> None:
        if self.is_deleting(chat_id):
            raise RuntimeError("Chat is being deleted")

    def open(self, chat_id: str, profile_id: str) -> ChatBinding:
        self._ensure_active(chat_id)
        if profile_id not in self._available_profiles:
            raise ValueError(f"Unknown Model Profile: {profile_id}")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._get_row(connection, chat_id)
            if row is None:
                connection.execute(
                    """INSERT INTO chat_bindings(
                           chat_id, profile_id, memory_thread_id
                       ) VALUES (?, ?, ?)""",
                    (chat_id, profile_id, chat_id),
                )
                connection.execute(
                    """INSERT INTO chat_memory_threads(chat_id, thread_id)
                       VALUES (?, ?)""",
                    (chat_id, chat_id),
                )
            else:
                current_profile = str(row[0])
                if (
                    current_profile in self._available_profiles
                    and current_profile != profile_id
                ):
                    raise ValueError(
                        "Model Profile cannot change inside an existing Chat"
                    )
                if current_profile != profile_id:
                    connection.execute(
                        "UPDATE chat_bindings SET profile_id = ? WHERE chat_id = ?",
                        (profile_id, chat_id),
                    )
            persisted = self._get_row(connection, chat_id)

        if persisted is None:  # pragma: no cover - guarded by the transaction above
            raise RuntimeError("Chat binding was not persisted")
        return self._binding(persisted)

    def select_mode(self, chat_id: str, mode: AgentMode) -> ChatBinding:
        self._ensure_active(chat_id)
        requested = AgentMode(mode)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._get_row(connection, chat_id)
            if row is None:
                raise KeyError(chat_id)
            current = AgentMode(str(row[1]))
            if bool(row[2]) and current is not requested:
                raise ValueError("Agent Mode cannot change after the first Turn")
            if not bool(row[2]) and current is not requested:
                connection.execute(
                    "UPDATE chat_bindings SET agent_mode = ? WHERE chat_id = ?",
                    (requested.value, chat_id),
                )
            persisted = self._get_row(connection, chat_id)

        if persisted is None:  # pragma: no cover - guarded by the transaction above
            raise RuntimeError("Chat binding disappeared during mode selection")
        return self._binding(persisted)

    def lock(self, chat_id: str) -> ChatBinding:
        self._ensure_active(chat_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._get_row(connection, chat_id) is None:
                raise KeyError(chat_id)
            connection.execute(
                "UPDATE chat_bindings SET mode_locked = 1 WHERE chat_id = ?",
                (chat_id,),
            )
            persisted = self._get_row(connection, chat_id)

        if persisted is None:  # pragma: no cover - guarded by the transaction above
            raise RuntimeError("Chat binding disappeared while locking its mode")
        return self._binding(persisted)

    @staticmethod
    def _validate_memory_thread(chat_id: str, thread_id: str) -> None:
        if thread_id != chat_id and not thread_id.startswith(f"{chat_id}:"):
            raise ValueError("Agent Memory thread belongs to another Chat")

    def reserve_memory_thread(self, chat_id: str) -> str:
        self._ensure_active(chat_id)
        thread_id = f"{chat_id}:{uuid.uuid4().hex}"
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._get_row(connection, chat_id) is None:
                raise KeyError(chat_id)
            connection.execute(
                """INSERT INTO chat_memory_threads(chat_id, thread_id)
                   VALUES (?, ?)""",
                (chat_id, thread_id),
            )
        return thread_id

    def use_memory_thread(self, chat_id: str, thread_id: str) -> ChatBinding:
        self._ensure_active(chat_id)
        self._validate_memory_thread(chat_id, thread_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            tracked = connection.execute(
                """SELECT 1 FROM chat_memory_threads
                   WHERE chat_id = ? AND thread_id = ?""",
                (chat_id, thread_id),
            ).fetchone()
            if tracked is None:
                raise ValueError("Unknown Agent Memory thread")
            connection.execute(
                "UPDATE chat_bindings SET memory_thread_id = ? WHERE chat_id = ?",
                (thread_id, chat_id),
            )
            persisted = self._get_row(connection, chat_id)

        if persisted is None:  # pragma: no cover - guarded by the transaction above
            raise RuntimeError("Chat binding disappeared while selecting Agent Memory")
        return self._binding(persisted)

    def new_memory_thread(self, chat_id: str) -> ChatBinding:
        return self.use_memory_thread(chat_id, self.reserve_memory_thread(chat_id))

    def owns_memory_thread(self, chat_id: str, thread_id: str) -> bool:
        self._validate_memory_thread(chat_id, thread_id)
        with self._connect() as connection:
            row = connection.execute(
                """SELECT 1 FROM chat_memory_threads
                   WHERE chat_id = ? AND thread_id = ?""",
                (chat_id, thread_id),
            ).fetchone()
        return row is not None

    def memory_threads(self, chat_id: str) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT thread_id FROM chat_memory_threads
                   WHERE chat_id = ? ORDER BY thread_id""",
                (chat_id,),
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def delete(self, chat_id: str) -> None:
        self.mark_deleting(chat_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM chat_memory_threads WHERE chat_id = ?", (chat_id,)
            )
            connection.execute(
                "DELETE FROM chat_bindings WHERE chat_id = ?", (chat_id,)
            )
