"""Persist the current model context and immutable pre-Turn snapshots.

The ReAct graph is stateless between runs. Only its messages (including the
rolling summary) cross this seam; graph nodes, tools and counters do not.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


class AgentMemory:
    def __init__(self, database: Path) -> None:
        database.parent.mkdir(parents=True, exist_ok=True)
        self.database = database
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS agent_context (
                    chat_id TEXT PRIMARY KEY, messages TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_snapshots (
                    id TEXT PRIMARY KEY, chat_id TEXT NOT NULL, messages TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS agent_snapshots_chat
                    ON agent_snapshots(chat_id);
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.database, timeout=30)) as db, db:
            yield db

    @staticmethod
    def _encode(messages: list[BaseMessage]) -> str:
        return json.dumps(messages_to_dict(messages), ensure_ascii=False)

    def load(self, chat_id: str) -> list[BaseMessage] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT messages FROM agent_context WHERE chat_id=?", (chat_id,)
            ).fetchone()
        return messages_from_dict(json.loads(row[0])) if row else None

    def save(self, chat_id: str, messages: list[BaseMessage]) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO agent_context VALUES (?, ?)",
                (chat_id, self._encode(messages)),
            )

    def checkpoint(self, chat_id: str, messages: list[BaseMessage]) -> str:
        snapshot_id = uuid.uuid4().hex
        with self._connect() as db:
            db.execute(
                "INSERT INTO agent_snapshots VALUES (?, ?, ?)",
                (snapshot_id, chat_id, self._encode(messages)),
            )
        return json.dumps({"version": 4, "chat_id": chat_id, "snapshot": snapshot_id})

    def restore(self, chat_id: str, token: dict) -> None:
        if token.get("chat_id") != chat_id:
            raise ValueError("Agent checkpoint belongs to another Chat")
        with self._connect() as db:
            row = db.execute(
                "SELECT messages FROM agent_snapshots WHERE id=? AND chat_id=?",
                (token.get("snapshot"), chat_id),
            ).fetchone()
            if row is None:
                raise ValueError("Agent checkpoint does not exist")
            db.execute(
                "INSERT OR REPLACE INTO agent_context VALUES (?, ?)", (chat_id, row[0])
            )

    def delete(self, chat_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM agent_context WHERE chat_id=?", (chat_id,))
            db.execute("DELETE FROM agent_snapshots WHERE chat_id=?", (chat_id,))
