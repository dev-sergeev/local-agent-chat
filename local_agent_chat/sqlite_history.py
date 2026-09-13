from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

from .runtime import HistorySnapshot, Turn


@dataclass(frozen=True)
class _StoredTurn:
    turn: Turn
    sequence: int
    created_at: str


@dataclass(frozen=True)
class _HistorySnapshotPayload:
    chat_id: str
    start_sequence: int
    turns: tuple[_StoredTurn, ...]


class SQLiteHistory:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    memory_checkpoint TEXT NOT NULL,
                    sandbox_snapshot TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(chat_id, sequence)
                );
                DROP TABLE IF EXISTS superseded_turns;
                """
            )
            self._migrate_created_at(connection)
            self._remove_search_index(connection)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA secure_delete = ON")
            yield connection

    @staticmethod
    def _migrate_created_at(connection: sqlite3.Connection) -> None:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(turns)")
        }
        if "created_at" not in columns:
            connection.execute("ALTER TABLE turns ADD COLUMN created_at TEXT")
            connection.execute(
                "UPDATE turns SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
            )

    @staticmethod
    def _remove_search_index(connection: sqlite3.Connection) -> None:
        # Cross-chat retrieval is no longer part of the sandbox-only agent.
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        ).fetchall():
            name = row[0]
            if name.startswith(("turns_search_", "turn_search_documents_")):
                connection.execute('DROP TRIGGER "' + name.replace('"', '""') + '"')
        connection.execute("DROP TABLE IF EXISTS turn_search_fts")
        connection.execute("DROP TABLE IF EXISTS turn_search_documents")

    async def append(self, turn: Turn) -> None:
        with self._connect() as connection:
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM turns WHERE chat_id = ?",
                (turn.chat_id,),
            ).fetchone()[0]
            self._insert(connection, turn, sequence)

    async def replace_from(self, turn_id: str, turn: Turn) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT chat_id, sequence FROM turns WHERE id = ?", (turn_id,)
            ).fetchone()
            if row is None:
                raise KeyError(turn_id)
            connection.execute(
                "DELETE FROM turns WHERE chat_id = ? AND sequence >= ?",
                (row["chat_id"], row["sequence"]),
            )
            self._insert(connection, turn, row["sequence"])

    async def snapshot_from(self, turn_id: str) -> HistorySnapshot:
        """Capture one active continuation for compensating a late UI failure."""

        with self._connect() as connection:
            root = connection.execute(
                "SELECT chat_id, sequence FROM turns WHERE id = ?", (turn_id,)
            ).fetchone()
            if root is None:
                raise KeyError(turn_id)
            rows = connection.execute(
                """SELECT * FROM turns
                   WHERE chat_id = ? AND sequence >= ?
                   ORDER BY sequence""",
                (root["chat_id"], root["sequence"]),
            ).fetchall()
        stored = tuple(
            _StoredTurn(
                turn=Turn(
                    id=row["id"],
                    chat_id=row["chat_id"],
                    text=row["text"],
                    answer=row["answer"],
                    memory_checkpoint=row["memory_checkpoint"],
                    sandbox_snapshot=row["sandbox_snapshot"],
                ),
                sequence=row["sequence"],
                created_at=row["created_at"],
            )
            for row in rows
        )
        return HistorySnapshot(
            _HistorySnapshotPayload(
                chat_id=root["chat_id"],
                start_sequence=root["sequence"],
                turns=stored,
            )
        )

    async def restore_snapshot(self, snapshot: HistorySnapshot) -> None:
        """Restore the exact active continuation after a failed Revision."""

        payload = snapshot.payload
        if not isinstance(payload, _HistorySnapshotPayload):
            raise TypeError("History snapshot was created by a different repository")
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM turns WHERE chat_id = ? AND sequence >= ?",
                (payload.chat_id, payload.start_sequence),
            )
            connection.executemany(
                """INSERT INTO turns
                   (id, chat_id, sequence, text, answer, memory_checkpoint,
                    sandbox_snapshot, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        stored.turn.id,
                        stored.turn.chat_id,
                        stored.sequence,
                        stored.turn.text,
                        stored.turn.answer,
                        stored.turn.memory_checkpoint,
                        stored.turn.sandbox_snapshot,
                        stored.created_at,
                    )
                    for stored in payload.turns
                ],
            )

    async def set_answer(self, turn_id: str, answer: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE turns SET answer = ? WHERE id = ?", (answer, turn_id)
            )
            if cursor.rowcount != 1:
                raise KeyError(turn_id)

    async def get(self, turn_id: str) -> Turn:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM turns WHERE id = ?", (turn_id,)
            ).fetchone()
        if row is None:
            raise KeyError(turn_id)
        return Turn(
            id=row["id"],
            chat_id=row["chat_id"],
            text=row["text"],
            answer=row["answer"],
            memory_checkpoint=row["memory_checkpoint"],
            sandbox_snapshot=row["sandbox_snapshot"],
        )

    async def has_chat(self, chat_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM turns WHERE chat_id = ? LIMIT 1", (chat_id,)
            ).fetchone()
        return row is not None

    async def delete_chat(self, chat_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM turns WHERE chat_id = ?", (chat_id,))

    async def context_messages(
        self, chat_id: str, *, before_checkpoint: str | None = None
    ):
        """Import current visible history, or the prefix before a legacy Turn.

        Never deserialize or execute obsolete graph/tool state. A missing
        legacy checkpoint is an error, not permission to restore other history.
        """
        from langchain_core.messages import AIMessage, HumanMessage

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM turns WHERE chat_id=? ORDER BY sequence", (chat_id,)
            ).fetchall()
        target = (
            json.loads(before_checkpoint) if before_checkpoint is not None else None
        )
        messages = []
        for row in rows:
            if target is not None and json.loads(row["memory_checkpoint"]) == target:
                return messages
            messages.extend(
                [
                    HumanMessage(content=row["text"], id=row["id"]),
                    AIMessage(content=row["answer"], id="answer-" + row["id"]),
                ]
            )
        if target is not None:
            raise ValueError(
                "Legacy Agent checkpoint does not belong to an active Turn"
            )
        return messages

    @staticmethod
    def _insert(connection: sqlite3.Connection, turn: Turn, sequence: int) -> None:
        connection.execute(
            """INSERT INTO turns
               (id, chat_id, sequence, text, answer, memory_checkpoint,
                sandbox_snapshot, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (
                turn.id,
                turn.chat_id,
                sequence,
                turn.text,
                turn.answer,
                turn.memory_checkpoint,
                turn.sandbox_snapshot,
            ),
        )
