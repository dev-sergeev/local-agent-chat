import sqlite3
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from local_agent_chat.agent_modes import AgentMode
from local_agent_chat.chat_bindings import ChatBinding, ChatBindings


def test_binding_is_an_immutable_slots_value(tmp_path: Path) -> None:
    bindings = ChatBindings(tmp_path / "bindings.sqlite3", ("local",))

    binding = bindings.open("chat-1", "local")

    assert not hasattr(binding, "__dict__")
    with pytest.raises(FrozenInstanceError):
        binding.mode_locked = True  # type: ignore[misc]


def test_binding_lifecycle_survives_restart(tmp_path: Path) -> None:
    database = tmp_path / "bindings.sqlite3"
    bindings = ChatBindings(database, ("local",))
    assert bindings.get("chat-1") is None

    opened = bindings.open("chat-1", "local")
    selected = bindings.select_mode("chat-1", AgentMode.EXTENDED)
    locked = bindings.lock("chat-1")
    renewed = bindings.new_memory_thread("chat-1")

    assert opened == ChatBinding("local", AgentMode.READ_ONLY, False, "chat-1")
    assert selected == ChatBinding("local", AgentMode.EXTENDED, False, "chat-1")
    assert locked == ChatBinding("local", AgentMode.EXTENDED, True, "chat-1")
    assert renewed.profile_id == "local"
    assert renewed.mode is AgentMode.EXTENDED
    assert renewed.mode_locked is True
    assert renewed.memory_thread_id.startswith("chat-1:")
    assert len(renewed.memory_thread_id) == len("chat-1:") + 32
    assert set(bindings.memory_threads("chat-1")) == {
        "chat-1",
        renewed.memory_thread_id,
    }

    reopened = ChatBindings(database, ("local",))
    assert reopened.get("chat-1") == renewed


def test_available_profile_is_immutable(tmp_path: Path) -> None:
    bindings = ChatBindings(tmp_path / "bindings.sqlite3", ("first", "second"))
    bindings.open("chat-1", "first")

    with pytest.raises(ValueError, match="cannot change"):
        bindings.open("chat-1", "second")

    assert bindings.get("chat-1").profile_id == "first"  # type: ignore[union-attr]


def test_stale_profile_can_be_replaced_by_an_available_profile(
    tmp_path: Path,
) -> None:
    database = tmp_path / "bindings.sqlite3"
    ChatBindings(database, ("removed",)).open("chat-1", "removed")

    current = ChatBindings(database, ("current",))
    migrated = current.open("chat-1", "current")

    assert migrated.profile_id == "current"
    assert ChatBindings(database, ("current",)).get("chat-1") == migrated


def test_unknown_profile_is_rejected_without_creating_a_chat(tmp_path: Path) -> None:
    bindings = ChatBindings(tmp_path / "bindings.sqlite3", ("local",))

    with pytest.raises(ValueError, match="Unknown Model Profile"):
        bindings.open("chat-1", "missing")

    assert bindings.get("chat-1") is None


def test_locked_mode_is_idempotent_but_cannot_change(tmp_path: Path) -> None:
    bindings = ChatBindings(tmp_path / "bindings.sqlite3", ("local",))
    bindings.open("chat-1", "local")
    bindings.select_mode("chat-1", AgentMode.EXTENDED)
    locked = bindings.lock("chat-1")

    assert bindings.lock("chat-1") == locked
    assert bindings.select_mode("chat-1", AgentMode.EXTENDED) == locked
    with pytest.raises(ValueError, match="cannot change"):
        bindings.select_mode("chat-1", AgentMode.READ_ONLY)
    assert bindings.get("chat-1") == locked


def test_instances_observe_authoritative_sqlite_state(tmp_path: Path) -> None:
    database = tmp_path / "bindings.sqlite3"
    first = ChatBindings(database, ("local",))
    second = ChatBindings(database, ("local",))
    first.open("chat-1", "local")
    assert first.get("chat-1").mode is AgentMode.READ_ONLY  # type: ignore[union-attr]

    second.select_mode("chat-1", AgentMode.EXTENDED)

    assert first.get("chat-1").mode is AgentMode.EXTENDED  # type: ignore[union-attr]


def test_new_memory_thread_preserves_chat_configuration(tmp_path: Path) -> None:
    bindings = ChatBindings(tmp_path / "bindings.sqlite3", ("local",))
    bindings.open("chat-1", "local")
    bindings.select_mode("chat-1", AgentMode.EXTENDED)
    before = bindings.lock("chat-1")

    first = bindings.new_memory_thread("chat-1")
    second = bindings.new_memory_thread("chat-1")

    assert (first.profile_id, first.mode, first.mode_locked) == (
        before.profile_id,
        before.mode,
        before.mode_locked,
    )
    assert first.memory_thread_id != before.memory_thread_id
    assert second.memory_thread_id != first.memory_thread_id


def test_only_a_tracked_memory_thread_can_be_activated(tmp_path: Path) -> None:
    bindings = ChatBindings(tmp_path / "bindings.sqlite3", ("local",))
    bindings.open("chat-1", "local")
    reserved = bindings.reserve_memory_thread("chat-1")

    assert bindings.use_memory_thread("chat-1", reserved).memory_thread_id == reserved
    with pytest.raises(ValueError, match="Unknown Agent Memory thread"):
        bindings.use_memory_thread("chat-1", "chat-1:untracked")
    with pytest.raises(ValueError, match="another Chat"):
        bindings.use_memory_thread("chat-1", "chat-2:foreign")


def test_delete_removes_the_persisted_binding(tmp_path: Path) -> None:
    database = tmp_path / "bindings.sqlite3"
    bindings = ChatBindings(database, ("local",))
    bindings.open("chat-1", "local")

    bindings.delete("chat-1")
    bindings.delete("chat-1")

    assert bindings.get("chat-1") is None
    assert bindings.memory_threads("chat-1") == ()
    assert bindings.is_deleting("chat-1") is True
    with pytest.raises(RuntimeError, match="being deleted"):
        bindings.open("chat-1", "local")
    assert ChatBindings(database, ("local",)).get("chat-1") is None


def test_missing_chat_mutations_fail_without_creating_state(tmp_path: Path) -> None:
    bindings = ChatBindings(tmp_path / "bindings.sqlite3", ("local",))

    for mutate in (
        lambda: bindings.select_mode("missing", AgentMode.EXTENDED),
        lambda: bindings.lock("missing"),
        lambda: bindings.new_memory_thread("missing"),
    ):
        with pytest.raises(KeyError):
            mutate()
    assert bindings.get("missing") is None


def test_legacy_schema_without_mode_migrates_to_locked_extended(
    tmp_path: Path,
) -> None:
    database = tmp_path / "bindings.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """CREATE TABLE active_branches (
                   chat_id TEXT PRIMARY KEY,
                   profile_id TEXT NOT NULL,
                   checkpoint_ns TEXT NOT NULL
               )"""
        )
        connection.execute(
            "INSERT INTO active_branches VALUES ('legacy', 'local', 'old-branch')"
        )

    migrated = ChatBindings(database, ("local",))

    assert migrated.get("legacy") == ChatBinding(
        "local", AgentMode.EXTENDED, True, "legacy"
    )
    assert ChatBindings(database, ("local",)).get("legacy") == migrated.get("legacy")


@pytest.mark.parametrize("mode_locked", [0, 1])
def test_invalid_mode_fails_closed_without_changing_its_lock(
    tmp_path: Path, mode_locked: int
) -> None:
    database = tmp_path / f"bindings-{mode_locked}.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """CREATE TABLE active_branches (
                   chat_id TEXT PRIMARY KEY,
                   profile_id TEXT NOT NULL,
                   checkpoint_ns TEXT NOT NULL,
                   agent_mode TEXT NOT NULL,
                   mode_locked INTEGER NOT NULL
               )"""
        )
        connection.execute(
            "INSERT INTO active_branches VALUES (?, ?, ?, ?, ?)",
            ("chat-1", "local", "branch", "unknown", mode_locked),
        )

    migrated = ChatBindings(database, ("local",))

    assert migrated.get("chat-1") == ChatBinding(
        "local", AgentMode.READ_ONLY, bool(mode_locked), "chat-1"
    )
    assert ChatBindings(database, ("local",)).get("chat-1") == migrated.get("chat-1")
