"""SQLite connections and transaction ownership for the opt-in NFS mode.

The CLI still owns exclusion between processes. Within that process every
store for the same resolved path shares a gate, including schema setup and
Chainlit sessions. A gate covers reads, commits, rollback and connection cleanup.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, closing, contextmanager, nullcontext
from pathlib import Path

from sqlalchemy import URL, event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncSessionTransaction,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import AsyncAdaptedQueuePool

_gates: dict[Path, threading.Lock] = {}
_registry_lock = threading.Lock()


def _nfs_enabled() -> bool:
    value = os.environ.get("LOCALCHAT_SQLITE_NOLOCK", "0").strip()
    if value not in {"0", "1"}:
        raise ValueError("LOCALCHAT_SQLITE_NOLOCK must be 0 or 1")
    return value == "1"


async def _finish_io(awaitable):
    """Cancellation cannot stop SQLite's worker; drain it before releasing ownership."""
    task = asyncio.ensure_future(awaitable)
    cancelled = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as error:
            if task.cancelled():
                raise
            cancelled = error
    if cancelled is not None:
        raise cancelled
    return result


class _NFSTransaction(AsyncSessionTransaction):
    async def __aexit__(self, *args):
        await _finish_io(super().__aexit__(*args))


class _NFSSession(AsyncSession):
    async def execute(self, *args, **kwargs):
        return await _finish_io(super().execute(*args, **kwargs))

    async def connection(self, *args, **kwargs):
        return await _finish_io(super().connection(*args, **kwargs))

    async def commit(self):
        await _finish_io(super().commit())

    async def rollback(self):
        await _finish_io(super().rollback())

    def begin(self):
        return _NFSTransaction(self)


def _configure_nfs_connection(connection) -> None:
    cursor = connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=DELETE")
        mode = cursor.fetchone()[0]
        if mode.lower() != "delete":
            raise ValueError("NFS SQLite requires journal_mode=DELETE")
        # EXTRA also syncs the directory when the DELETE journal is removed.
        cursor.execute("PRAGMA synchronous=EXTRA")
        cursor.execute("PRAGMA synchronous")
        if cursor.fetchone()[0] != 3:
            raise ValueError("NFS SQLite requires synchronous=EXTRA")
    finally:
        cursor.close()


class SQLiteDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        # Read after CLI dotenv loading, and keep the mode stable for this store.
        self.nolock = _nfs_enabled()
        with _registry_lock:
            self._gate = _gates.setdefault(self.path, threading.Lock())

    @property
    def url(self) -> URL:
        if self.nolock:
            return URL.create(
                "sqlite+aiosqlite",
                database=self.path.as_uri(),
                query={"nolock": "1", "uri": "true"},
            )
        return URL.create("sqlite+aiosqlite", database=str(self.path))

    def _check_journal(self) -> None:
        # nolock cannot safely open/migrate WAL. Inspect its persistent header
        # before SQLite could touch the WAL or shared-memory files.
        try:
            with self.path.open("rb") as source:
                header = source.read(20)
        except FileNotFoundError:
            return
        if header.startswith(b"SQLite format 3\x00") and 2 in header[18:20]:
            raise ValueError(
                f"NFS SQLite cannot open WAL database {self.path}. "
                "Stop all instances and convert it to journal_mode=DELETE "
                "on storage with working SQLite locks before enabling "
                "LOCALCHAT_SQLITE_NOLOCK=1."
            )

    @contextmanager
    def connect(self, *, timeout: float = 5) -> Iterator[sqlite3.Connection]:
        with self._gate if self.nolock else nullcontext():
            if self.nolock:
                self._check_journal()
            target = self.path.as_uri() + "?nolock=1" if self.nolock else str(self.path)
            with closing(
                sqlite3.connect(target, timeout=timeout, uri=self.nolock)
            ) as connection:
                if self.nolock:
                    _configure_nfs_connection(connection)
                with connection:
                    yield connection

    def serialized_sessions(self):
        """Build the Chainlit engine and guard all inherited/custom SQL paths."""
        engine = create_async_engine(
            self.url, poolclass=AsyncAdaptedQueuePool, pool_size=1, max_overflow=0
        )

        @event.listens_for(engine.sync_engine, "connect")
        def configure(connection, _record):
            _configure_nfs_connection(connection)

        factory = async_sessionmaker(engine, class_=_NFSSession, expire_on_commit=False)

        @asynccontextmanager
        async def session() -> AsyncIterator[AsyncSession]:
            # A threading gate also coordinates synchronous stores and multiple
            # layers/event loops. Poll without blocking the loop or abandoning
            # a worker that could acquire the gate after task cancellation.
            while not self._gate.acquire(blocking=False):
                await asyncio.sleep(0.01)
            try:
                self._check_journal()
                current = factory()
                try:
                    yield current
                finally:
                    await _finish_io(current.close())
            finally:
                self._gate.release()

        return engine, session
