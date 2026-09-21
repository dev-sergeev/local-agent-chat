from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .sqlite_storage import SQLiteDatabase


@dataclass(frozen=True, slots=True)
class ChatBinding:
    profile_id: str


class ChatBindings:
    """The persisted model choice for each Chat; no capability modes or branches."""

    def __init__(self, database: Path, available_profile_ids: Iterable[str]) -> None:
        self._profiles = tuple(dict.fromkeys(available_profile_ids))
        if not self._profiles or any(not p for p in self._profiles):
            raise ValueError("At least one nonempty Model Profile must be available")
        self._database = database
        self._sqlite = SQLiteDatabase(database)
        self._deleting: set[str] = set()
        database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS chat_profiles (chat_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL)"
            )
            for table in ("chat_bindings", "active_branches"):
                if db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone():
                    db.execute(
                        f"INSERT OR IGNORE INTO chat_profiles SELECT chat_id, profile_id FROM {table}"
                    )
                    db.execute(f"DROP TABLE {table}")
            db.execute("DROP TABLE IF EXISTS chat_memory_threads")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._sqlite.connect(timeout=30) as db:
            yield db

    def get(self, chat_id: str) -> ChatBinding | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT profile_id FROM chat_profiles WHERE chat_id=?", (chat_id,)
            ).fetchone()
        return ChatBinding(row[0]) if row else None

    def is_deleting(self, chat_id: str) -> bool:
        return chat_id in self._deleting

    def mark_deleting(self, chat_id: str) -> None:
        self._deleting.add(chat_id)

    def open(self, chat_id: str, *profile_hints: str | None) -> ChatBinding:
        if self.is_deleting(chat_id):
            raise RuntimeError("Chat is being deleted")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT profile_id FROM chat_profiles WHERE chat_id=?", (chat_id,)
            ).fetchone()
            # A valid persisted profile is immutable. Removed profiles fall back
            # to an available hint or the configured default on resume.
            profile = (
                row[0]
                if row and row[0] in self._profiles
                else next(
                    (hint for hint in profile_hints if hint in self._profiles),
                    self._profiles[0],
                )
            )
            db.execute(
                "INSERT OR REPLACE INTO chat_profiles VALUES (?, ?)", (chat_id, profile)
            )
        return ChatBinding(profile)

    def delete(self, chat_id: str) -> None:
        self.mark_deleting(chat_id)
        with self._connect() as db:
            db.execute("DELETE FROM chat_profiles WHERE chat_id=?", (chat_id,))
