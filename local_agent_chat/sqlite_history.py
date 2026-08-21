from __future__ import annotations

import sqlite3
from pathlib import Path

from .runtime import Turn


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
                    UNIQUE(chat_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS superseded_turns (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    memory_checkpoint TEXT NOT NULL,
                    sandbox_snapshot TEXT NOT NULL,
                    superseded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection

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
                """INSERT INTO superseded_turns
                   (id, chat_id, sequence, text, answer, memory_checkpoint, sandbox_snapshot)
                   SELECT id, chat_id, sequence, text, answer, memory_checkpoint, sandbox_snapshot
                   FROM turns WHERE chat_id = ? AND sequence >= ?""",
                (row["chat_id"], row["sequence"]),
            )
            connection.execute(
                "DELETE FROM turns WHERE chat_id = ? AND sequence >= ?",
                (row["chat_id"], row["sequence"]),
            )
            self._insert(connection, turn, row["sequence"])

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

    async def delete_chat(self, chat_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM turns WHERE chat_id = ?", (chat_id,))
            connection.execute(
                "DELETE FROM superseded_turns WHERE chat_id = ?", (chat_id,)
            )

    @staticmethod
    def _insert(connection: sqlite3.Connection, turn: Turn, sequence: int) -> None:
        connection.execute(
            """INSERT INTO turns
               (id, chat_id, sequence, text, answer, memory_checkpoint, sandbox_snapshot)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
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
