"""Exercise transaction boundaries, not just the nolock URI spelling."""

import asyncio
import os
import select
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from sqlalchemy import text

from local_agent_chat.agent_memory import AgentMemory
from local_agent_chat.chainlit_data import create_chainlit_data_layer
from local_agent_chat.chat_bindings import ChatBindings
from local_agent_chat.installation import runtime_workspace
from local_agent_chat.runtime import Turn
from local_agent_chat.sqlite_history import SQLiteHistory
from local_agent_chat.sqlite_storage import SQLiteDatabase


@pytest.fixture
def nfs_mode(monkeypatch):
    # The modules above were imported before the CLI would load .env.
    monkeypatch.setenv("LOCALCHAT_SQLITE_NOLOCK", "1")


async def test_chainlit_serializes_transactions_across_layers(tmp_path, nfs_mode):
    path = tmp_path / "история ?#%.sqlite3"
    first = create_chainlit_data_layer(path)
    second = create_chainlit_data_layer(path)
    entered = asyncio.Event()
    started = asyncio.Event()

    async def reader():
        started.set()
        async with second.async_session() as session:
            rows = await session.execute(text("SELECT id FROM users"))
            entered.set()
            return list(rows.scalars())

    try:
        async with first.async_session() as session:
            await session.execute(
                text(
                    "INSERT INTO users (id, identifier, \"createdAt\") VALUES ('kept', 'local', 'today')"
                )
            )
            task = asyncio.create_task(reader())
            await started.wait()
            await asyncio.sleep(0.05)
            assert not entered.is_set(), "a reader entered an unfinished write"
            await session.commit()
        assert await asyncio.wait_for(task, 3) == ["kept"]
        async with second.async_session() as session:
            assert (
                await session.execute(text("PRAGMA journal_mode"))
            ).scalar() == "delete"
            assert (await session.execute(text("PRAGMA synchronous"))).scalar() == 3
            assert (
                await session.execute(text("PRAGMA integrity_check"))
            ).scalar() == "ok"
    finally:
        await first.close()
        await second.close()


async def test_cancelled_writer_rolls_back_before_next_session(tmp_path, nfs_mode):
    layer = create_chainlit_data_layer(tmp_path / "chainlit.sqlite3")
    written = asyncio.Event()

    async def writer():
        async with layer.async_session() as session:
            await session.execute(
                text(
                    "INSERT INTO users (id, identifier, \"createdAt\") VALUES ('aborted', 'local', 'today')"
                )
            )
            written.set()
            await asyncio.Event().wait()

    try:
        task = asyncio.create_task(writer())
        await asyncio.wait_for(written.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with asyncio.timeout(3):
            async with layer.async_session() as session:
                assert (
                    await session.execute(text("SELECT count(*) FROM users"))
                ).scalar() == 0
                await session.execute(
                    text(
                        "INSERT INTO users (id, identifier, \"createdAt\") VALUES ('after', 'local', 'today')"
                    )
                )
                await session.commit()
    finally:
        await layer.close()


async def test_repeated_cancel_waits_for_sqlite_worker(tmp_path, nfs_mode):
    layer = create_chainlit_data_layer(tmp_path / "chainlit.sqlite3")
    running = threading.Event()
    finish = threading.Event()

    def slow_write():
        running.set()
        assert finish.wait(5)
        return "pending"

    async def writer():
        async with layer.async_session() as session:
            connection = await session.connection()
            await connection.run_sync(
                lambda conn: conn.connection.create_function(
                    "slow_write", 0, slow_write
                )
            )
            await session.execute(
                text(
                    'INSERT INTO users (id, identifier, "createdAt") '
                    "VALUES ('aborted', slow_write(), 'today')"
                )
            )
            await session.commit()

    task = asyncio.create_task(writer())
    try:
        assert await asyncio.to_thread(running.wait, 3)
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done(), "cancellation abandoned an active SQLite worker"
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        async with layer.async_session() as session:
            assert (
                await session.execute(text("SELECT count(*) FROM users"))
            ).scalar() == 0
            assert (
                await session.execute(text("PRAGMA integrity_check"))
            ).scalar() == "ok"
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)
        await layer.close()


def test_memory_and_profiles_share_transaction_guard(tmp_path, nfs_mode):
    path = tmp_path / "checkpoints.sqlite3"
    memory = AgentMemory(path)
    bindings = ChatBindings(path, ["test"])
    entered = threading.Event()
    started = threading.Event()

    def read_profile():
        started.set()
        with bindings._connect() as connection:
            entered.set()
            return connection.execute("SELECT messages FROM agent_context").fetchone()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with memory._connect() as connection:
            connection.execute("INSERT INTO agent_context VALUES ('chat', '[]')")
            pending = executor.submit(read_profile)
            assert started.wait(3)
            assert not entered.wait(0.05), "a second store entered the same database"
        assert pending.result(timeout=3) == ("[]",)
    assert bindings.open("chat").profile_id == "test"
    assert memory.load("chat") == []


def test_nfs_mode_rejects_existing_wal_without_changing_it(tmp_path, nfs_mode):
    path = tmp_path / "checkpoints.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE preserved (value TEXT)")
        connection.execute("INSERT INTO preserved VALUES ('kept')")
    with pytest.raises(ValueError, match="WAL"):
        AgentMemory(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert connection.execute("SELECT * FROM preserved").fetchone() == ("kept",)


async def test_all_stores_work_when_regular_connections_are_unavailable(
    tmp_path, nfs_mode, monkeypatch
):
    connect = sqlite3.connect

    def locking_unavailable(database, *args, **kwargs):
        if not kwargs.get("uri") or "nolock=1" not in str(database):
            raise sqlite3.OperationalError("database is locked")
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", locking_unavailable)
    layer = create_chainlit_data_layer(tmp_path / "chainlit.sqlite3")
    memory = AgentMemory(tmp_path / "checkpoints.sqlite3")
    bindings = ChatBindings(tmp_path / "checkpoints.sqlite3", ["test"])
    history = SQLiteHistory(tmp_path / "runtime-history.sqlite3")
    try:
        memory.save("chat", [])
        assert memory.load("chat") == []
        assert bindings.open("chat").profile_id == "test"
        turn = Turn("turn", "chat", "request", "answer", "checkpoint", "snapshot")
        await history.append(turn)
        assert (await history.get("turn")).answer == "answer"
        await asyncio.gather(
            *(layer.update_thread(f"chat-{i}", name=f"Chat {i}") for i in range(20))
        )
        async with layer.async_session() as session:
            assert (
                await session.execute(text("SELECT count(*) FROM threads"))
            ).scalar() == 20
    finally:
        await layer.close()


async def test_cancelled_waiter_does_not_keep_database_ownership(tmp_path, nfs_mode):
    layer = create_chainlit_data_layer(tmp_path / "chainlit.sqlite3")
    started = asyncio.Event()

    async def waiting():
        started.set()
        async with layer.async_session():
            pytest.fail("waiter entered before the owner finished")

    try:
        async with layer.async_session():
            task = asyncio.create_task(waiting())
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        async with asyncio.timeout(3):
            async with layer.async_session() as session:
                assert (await session.execute(text("SELECT 1"))).scalar() == 1
    finally:
        await layer.close()


@pytest.mark.parametrize("setting", [None, "0"])
def test_default_mode_keeps_sqlite_locking_and_sync_defaults(
    tmp_path, monkeypatch, setting
):
    if setting is None:
        monkeypatch.delenv("LOCALCHAT_SQLITE_NOLOCK", raising=False)
    else:
        monkeypatch.setenv("LOCALCHAT_SQLITE_NOLOCK", setting)
    path = tmp_path / "state.sqlite3"
    store = SQLiteDatabase(path)
    with store.connect() as first:
        first.execute("CREATE TABLE example (value INTEGER)")
        first.execute("BEGIN IMMEDIATE")
        with sqlite3.connect(path, timeout=0) as second:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                second.execute("BEGIN IMMEDIATE")
            assert (
                first.execute("PRAGMA synchronous").fetchone()
                == second.execute("PRAGMA synchronous").fetchone()
            )


def test_invalid_mode_is_rejected_before_database_creation(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALCHAT_SQLITE_NOLOCK", "yes")
    path = tmp_path / "state.sqlite3"
    with pytest.raises(ValueError, match="LOCALCHAT_SQLITE_NOLOCK"):
        SQLiteDatabase(path)
    assert not path.exists()


def test_process_crash_preserves_commits_and_recovers_hot_journal(tmp_path, nfs_mode):
    _check_crash_recovery(tmp_path)


def _check_crash_recovery(tmp_path):
    script = """
import errno
import fcntl
import signal
import sys
from pathlib import Path
from local_agent_chat.installation import runtime_workspace
from local_agent_chat.sqlite_storage import SQLiteDatabase

def unavailable(*args):
    raise OSError(errno.ENOLCK, 'No locks available')
fcntl.flock = unavailable
directory = Path(sys.argv[1])
with runtime_workspace(directory):
    database = SQLiteDatabase(directory / 'crash.sqlite3')
    with database.connect() as connection:
        connection.execute('CREATE TABLE durable (id INTEGER PRIMARY KEY, value TEXT)')
        connection.executemany('INSERT INTO durable VALUES (?, ?)',
                               [(i, 'committed') for i in range(100)])
    with database.connect() as connection:
        connection.execute('PRAGMA cache_size=1')
        connection.execute('UPDATE durable SET value=?', ('uncommitted' * 1000,))
        print('ready', flush=True)
        signal.pause()
"""
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", script, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "LOCALCHAT_SQLITE_NOLOCK": "1"},
        text=True,
    )
    try:
        assert select.select([process.stdout], [], [], 10)[0], (
            "worker did not become ready"
        )
        assert process.stdout.readline().strip() == "ready"
        with pytest.raises(ValueError, match="Another LocalChat"):
            with runtime_workspace(tmp_path):
                pass
        process.kill()
        process.wait(timeout=5)
        guard = tmp_path / ".localchat.lock.d"
        with pytest.raises(ValueError, match="stale lock"):
            with runtime_workspace(tmp_path):
                pass
        assert guard.is_dir()
        # The test owns this process and has confirmed it exited; no automatic
        # stale-lock deletion is performed by production code.
        guard.rmdir()
        assert (tmp_path / "crash.sqlite3-journal").exists()
        with runtime_workspace(tmp_path):
            with SQLiteDatabase(tmp_path / "crash.sqlite3").connect() as connection:
                assert connection.execute("PRAGMA integrity_check").fetchone() == (
                    "ok",
                )
                assert connection.execute(
                    "SELECT count(*) FROM durable WHERE value='committed'"
                ).fetchone() == (100,)
                connection.execute("INSERT INTO durable VALUES (100, 'after restart')")
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


@pytest.mark.skipif(
    not os.environ.get("LOCALCHAT_TEST_NFS_DIR"),
    reason="set LOCALCHAT_TEST_NFS_DIR to an existing writable NFS directory",
)
async def test_real_nfs_round_trip_and_crash_recovery(nfs_mode):
    # Never open the user's real databases. Exercise only a fresh test directory.
    with TemporaryDirectory(
        prefix="localchat-storage-test-", dir=os.environ["LOCALCHAT_TEST_NFS_DIR"]
    ) as temporary:
        directory = Path(temporary)
        with runtime_workspace(directory):
            layer = create_chainlit_data_layer(directory / "chainlit.sqlite3")
            memory = AgentMemory(directory / "checkpoints.sqlite3")
            bindings = ChatBindings(directory / "checkpoints.sqlite3", ["test"])
            history = SQLiteHistory(directory / "runtime-history.sqlite3")
            try:
                memory.save("chat", [])
                bindings.open("chat")
                await history.append(Turn("turn", "chat", "q", "a", "cp", "snapshot"))
                await asyncio.gather(
                    *(layer.update_thread(f"chat-{i}", name=str(i)) for i in range(20))
                )
            finally:
                await layer.close()
        with runtime_workspace(directory):
            layer = create_chainlit_data_layer(directory / "chainlit.sqlite3")
            try:
                assert (await layer.get_thread("chat-19"))["name"] == "19"
                assert AgentMemory(directory / "checkpoints.sqlite3").load("chat") == []
                assert (
                    ChatBindings(directory / "checkpoints.sqlite3", ["test"])
                    .get("chat")
                    .profile_id
                    == "test"
                )
                assert (
                    await SQLiteHistory(directory / "runtime-history.sqlite3").get(
                        "turn"
                    )
                ).answer == "a"
            finally:
                await layer.close()
            for filename in (
                "chainlit.sqlite3",
                "checkpoints.sqlite3",
                "runtime-history.sqlite3",
            ):
                with SQLiteDatabase(directory / filename).connect() as connection:
                    assert connection.execute("PRAGMA integrity_check").fetchone() == (
                        "ok",
                    )
        await asyncio.to_thread(_check_crash_recovery, directory / "crash")
