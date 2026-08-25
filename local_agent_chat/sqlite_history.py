from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .runtime import Turn

DEFAULT_SEARCH_LIMIT = 5
MAX_SEARCH_LIMIT = 10
MAX_SEARCH_QUERY_CHARS = 500
MAX_SEARCH_TERMS = 16
MAX_CONTEXT_TURNS = 2


@dataclass(frozen=True)
class GlobalMemorySearchHit:
    chat_id: str
    turn_id: str
    sequence: int
    created_at: str
    user_snippet: str
    assistant_snippet: str


@dataclass(frozen=True)
class GlobalMemoryTurn:
    chat_id: str
    turn_id: str
    sequence: int
    created_at: str
    text: str
    answer: str
    selected: bool


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
            self._migrate_created_at(connection)
            self._create_search_index(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA secure_delete = ON")
        return connection

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
    def _create_search_index(connection: sqlite3.Connection) -> None:
        """Create and reconcile the derived FTS index with active Turns."""

        connection.executescript(
            """
            DROP TRIGGER IF EXISTS turn_search_documents_ad;
            DROP TRIGGER IF EXISTS turn_search_documents_au;

            CREATE TABLE IF NOT EXISTS turn_search_documents (
                document_id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id TEXT NOT NULL UNIQUE,
                chat_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                user_text TEXT NOT NULL,
                assistant_text TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS turn_search_fts USING fts5(
                user_text,
                assistant_text,
                content='turn_search_documents',
                content_rowid='document_id',
                tokenize='unicode61 remove_diacritics 2'
            );

            CREATE TRIGGER IF NOT EXISTS turn_search_documents_ai
            AFTER INSERT ON turn_search_documents BEGIN
                INSERT INTO turn_search_fts(rowid, user_text, assistant_text)
                VALUES (new.document_id, new.user_text, new.assistant_text);
            END;

            CREATE TRIGGER IF NOT EXISTS turn_search_documents_bd
            BEFORE DELETE ON turn_search_documents BEGIN
                DELETE FROM turn_search_fts WHERE rowid = old.document_id;
            END;

            CREATE TRIGGER IF NOT EXISTS turn_search_documents_bu
            BEFORE UPDATE ON turn_search_documents BEGIN
                DELETE FROM turn_search_fts WHERE rowid = old.document_id;
            END;

            CREATE TRIGGER IF NOT EXISTS turn_search_documents_au
            AFTER UPDATE ON turn_search_documents BEGIN
                INSERT INTO turn_search_fts(rowid, user_text, assistant_text)
                VALUES (new.document_id, new.user_text, new.assistant_text);
            END;

            CREATE TRIGGER IF NOT EXISTS turns_search_ai
            AFTER INSERT ON turns BEGIN
                INSERT INTO turn_search_documents(
                    turn_id, chat_id, sequence, created_at,
                    user_text, assistant_text
                ) VALUES (
                    new.id, new.chat_id, new.sequence, new.created_at,
                    new.text, new.answer
                );
            END;

            CREATE TRIGGER IF NOT EXISTS turns_search_ad
            AFTER DELETE ON turns BEGIN
                DELETE FROM turn_search_documents WHERE turn_id = old.id;
            END;

            CREATE TRIGGER IF NOT EXISTS turns_search_au
            AFTER UPDATE OF id, chat_id, sequence, created_at, text, answer ON turns BEGIN
                UPDATE turn_search_documents SET
                    turn_id = new.id,
                    chat_id = new.chat_id,
                    sequence = new.sequence,
                    created_at = new.created_at,
                    user_text = new.text,
                    assistant_text = new.answer
                WHERE turn_id = old.id;
            END;
            """
        )
        # Triggers do not backfill existing rows. Reconciliation also repairs an
        # interrupted/manual database edit without indexing the audit table.
        connection.execute(
            """DELETE FROM turn_search_documents
               WHERE turn_id NOT IN (SELECT id FROM turns)"""
        )
        connection.execute(
            """UPDATE turn_search_documents
               SET chat_id = (SELECT chat_id FROM turns WHERE id = turn_id),
                   sequence = (SELECT sequence FROM turns WHERE id = turn_id),
                   created_at = (SELECT created_at FROM turns WHERE id = turn_id),
                   user_text = (SELECT text FROM turns WHERE id = turn_id),
                   assistant_text = (SELECT answer FROM turns WHERE id = turn_id)
               WHERE EXISTS (
                   SELECT 1 FROM turns
                   WHERE turns.id = turn_search_documents.turn_id
                     AND (turns.chat_id != turn_search_documents.chat_id
                       OR turns.sequence != turn_search_documents.sequence
                       OR turns.created_at != turn_search_documents.created_at
                       OR turns.text != turn_search_documents.user_text
                       OR turns.answer != turn_search_documents.assistant_text)
               )"""
        )
        connection.execute(
            """INSERT INTO turn_search_documents(
                   turn_id, chat_id, sequence, created_at,
                   user_text, assistant_text
               )
               SELECT id, chat_id, sequence, created_at, text, answer
               FROM turns
               WHERE id NOT IN (SELECT turn_id FROM turn_search_documents)"""
        )
        connection.execute(
            "INSERT INTO turn_search_fts(turn_search_fts, rank) VALUES('secure-delete', 1)"
        )
        try:
            connection.execute(
                """INSERT INTO turn_search_fts(turn_search_fts, rank)
                   VALUES('integrity-check', 1)"""
            )
        except sqlite3.DatabaseError:
            # The FTS table is derived and can be deterministically repaired
            # from active Turns after an interrupted write or older trigger set.
            connection.execute(
                "INSERT INTO turn_search_fts(turn_search_fts) VALUES('rebuild')"
            )
            connection.execute(
                """INSERT INTO turn_search_fts(turn_search_fts, rank)
                   VALUES('secure-delete', 1)"""
            )
            connection.execute(
                """INSERT INTO turn_search_fts(turn_search_fts, rank)
                   VALUES('integrity-check', 1)"""
            )

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

    async def has_chat(self, chat_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM turns WHERE chat_id = ? LIMIT 1", (chat_id,)
            ).fetchone()
        return row is not None

    async def delete_chat(self, chat_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM turns WHERE chat_id = ?", (chat_id,))
            connection.execute(
                "DELETE FROM superseded_turns WHERE chat_id = ?", (chat_id,)
            )

    async def search_past_chats(
        self,
        query: str,
        *,
        exclude_chat_id: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[GlobalMemorySearchHit]:
        match_query = self._match_query(query)
        if not match_query:
            return []
        bounded_limit = max(
            1,
            min(self._coerce_int(limit, DEFAULT_SEARCH_LIMIT), MAX_SEARCH_LIMIT),
        )
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT documents.chat_id, documents.turn_id,
                          documents.sequence, documents.created_at,
                          snippet(turn_search_fts, 0, '[', ']', ' … ', 24)
                              AS user_snippet,
                          snippet(turn_search_fts, 1, '[', ']', ' … ', 24)
                              AS assistant_snippet
                   FROM turn_search_fts
                   JOIN turn_search_documents AS documents
                     ON documents.document_id = turn_search_fts.rowid
                   WHERE turn_search_fts MATCH ?
                     AND documents.chat_id != ?
                   ORDER BY bm25(turn_search_fts, 3.0, 1.0),
                            documents.created_at DESC,
                            documents.document_id DESC
                   LIMIT ?""",
                (match_query, exclude_chat_id, bounded_limit),
            ).fetchall()
        return [
            GlobalMemorySearchHit(
                chat_id=row["chat_id"],
                turn_id=row["turn_id"],
                sequence=row["sequence"],
                created_at=row["created_at"],
                user_snippet=row["user_snippet"],
                assistant_snippet=row["assistant_snippet"],
            )
            for row in rows
        ]

    async def read_past_chat(
        self,
        chat_id: str,
        turn_id: str,
        *,
        exclude_chat_id: str,
        context_turns: int = 1,
    ) -> list[GlobalMemoryTurn]:
        if chat_id == exclude_chat_id:
            return []
        bounded_context = max(
            0, min(self._coerce_int(context_turns, 1), MAX_CONTEXT_TURNS)
        )
        with self._connect() as connection:
            selected = connection.execute(
                """SELECT sequence FROM turns
                   WHERE id = ? AND chat_id = ?""",
                (turn_id, chat_id),
            ).fetchone()
            if selected is None:
                return []
            sequence = selected["sequence"]
            rows = connection.execute(
                """SELECT id, chat_id, sequence, created_at, text, answer
                   FROM turns
                   WHERE chat_id = ? AND sequence BETWEEN ? AND ?
                   ORDER BY sequence""",
                (chat_id, sequence - bounded_context, sequence + bounded_context),
            ).fetchall()
        return [
            GlobalMemoryTurn(
                chat_id=row["chat_id"],
                turn_id=row["id"],
                sequence=row["sequence"],
                created_at=row["created_at"],
                text=row["text"],
                answer=row["answer"],
                selected=row["id"] == turn_id,
            )
            for row in rows
        ]

    @staticmethod
    def _match_query(query: object) -> str:
        if not isinstance(query, str):
            return ""
        clean = "".join(
            character if character.isprintable() else " "
            for character in query[:MAX_SEARCH_QUERY_CHARS]
        )
        terms = re.findall(r"[\w][\w./:@+\\-]*", clean, flags=re.UNICODE)
        quoted = [f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms]
        # Natural-language tool queries often contain one or two words that do
        # not occur verbatim in the source. OR keeps candidate recall useful;
        # BM25 still ranks Turns matching more and rarer terms first.
        return " OR ".join(quoted[:MAX_SEARCH_TERMS])

    @staticmethod
    def _coerce_int(value: object, default: int) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            return default

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
