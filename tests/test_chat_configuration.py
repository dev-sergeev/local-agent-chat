from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from local_agent_chat.agent_modes import AgentMode
from local_agent_chat.chat_bindings import ChatBindings
from local_agent_chat.chat_configuration import ChatConfigurations


async def no_persisted_request(_chat_id: str) -> bool:
    return False


def test_profile_priority_and_stale_profile_migration(tmp_path: Path) -> None:
    database = tmp_path / "bindings.sqlite3"
    old_bindings = ChatBindings(database, ("removed",))
    old_bindings.open("stale-hint", "removed")
    old_bindings.open("stale-default", "removed")

    bindings = ChatBindings(database, ("first", "second"))
    configurations = ChatConfigurations(
        bindings,
        ("first", "second"),
        no_persisted_request,
    )

    hinted = configurations.open("hinted", "missing", "second", "first")
    persisted = configurations.open("hinted", "first")
    defaulted = configurations.open("defaulted", "missing")
    migrated_to_hint = configurations.open("stale-hint", "missing", "second")
    migrated_to_default = configurations.open("stale-default", "missing")

    assert hinted.profile_id == "second"
    assert persisted.profile_id == "second"
    assert defaulted.profile_id == "first"
    assert migrated_to_hint.profile_id == "second"
    assert migrated_to_default.profile_id == "first"
    with pytest.raises(FrozenInstanceError):
        hinted.mode_locked = True  # type: ignore[misc]


def test_mode_selection_message_lock_and_locked_conflict(tmp_path: Path) -> None:
    bindings = ChatBindings(tmp_path / "bindings.sqlite3", ("local",))
    configurations = ChatConfigurations(
        bindings,
        ("local",),
        no_persisted_request,
    )

    selected = configurations.select_mode("chat-1", AgentMode.EXTENDED, "local")
    locked = configurations.accept_message("chat-1", "local")
    rejected = configurations.select_mode("chat-1", AgentMode.READ_ONLY, "local")

    assert selected.mode is AgentMode.EXTENDED
    assert selected.mode_locked is False
    assert locked.mode is AgentMode.EXTENDED
    assert locked.mode_locked is True
    assert rejected == locked
    assert configurations.current("chat-1") == locked


@pytest.mark.asyncio
@pytest.mark.parametrize("has_request", [False, True])
async def test_recovery_locks_only_with_persisted_request_evidence(
    tmp_path: Path, has_request: bool
) -> None:
    calls: list[str] = []

    async def evidence(chat_id: str) -> bool:
        calls.append(chat_id)
        return has_request

    bindings = ChatBindings(tmp_path / f"bindings-{has_request}.sqlite3", ("local",))
    configurations = ChatConfigurations(bindings, ("local",), evidence)
    configurations.select_mode("chat-1", AgentMode.EXTENDED, "local")

    recovered = await configurations.recover("chat-1", "local")

    assert calls == ["chat-1"]
    assert recovered.mode is AgentMode.EXTENDED
    assert recovered.mode_locked is has_request
    assert configurations.current("chat-1") == recovered


@pytest.mark.asyncio
async def test_recovery_evidence_failure_does_not_lock_mode(tmp_path: Path) -> None:
    async def failed_evidence(_chat_id: str) -> bool:
        raise RuntimeError("history unavailable")

    bindings = ChatBindings(tmp_path / "bindings.sqlite3", ("local",))
    configurations = ChatConfigurations(bindings, ("local",), failed_evidence)

    with pytest.raises(RuntimeError, match="history unavailable"):
        await configurations.recover("chat-1", "local")

    current = configurations.current("chat-1")
    assert current is not None
    assert current.mode is AgentMode.READ_ONLY
    assert current.mode_locked is False


@pytest.mark.asyncio
async def test_locked_chat_does_not_depend_on_recovery_evidence(tmp_path: Path) -> None:
    async def failed_evidence(_chat_id: str) -> bool:
        raise RuntimeError("history unavailable")

    bindings = ChatBindings(tmp_path / "bindings.sqlite3", ("local",))
    configurations = ChatConfigurations(bindings, ("local",), failed_evidence)
    locked = configurations.accept_message("chat-1", "local")

    assert await configurations.recover("chat-1", "local") == locked


@pytest.mark.asyncio
async def test_deletion_tombstone_blocks_every_interface_entry(
    tmp_path: Path,
) -> None:
    evidence_calls: list[str] = []

    async def evidence(chat_id: str) -> bool:
        evidence_calls.append(chat_id)
        return True

    bindings = ChatBindings(tmp_path / "bindings.sqlite3", ("local",))
    configurations = ChatConfigurations(bindings, ("local",), evidence)
    configurations.open("chat-1", "local")
    bindings.mark_deleting("chat-1")

    for operation in (
        lambda: configurations.current("chat-1"),
        lambda: configurations.open("chat-1", "local"),
        lambda: configurations.select_mode("chat-1", AgentMode.EXTENDED, "local"),
        lambda: configurations.accept_message("chat-1", "local"),
    ):
        with pytest.raises(RuntimeError, match="being deleted"):
            operation()

    with pytest.raises(RuntimeError, match="being deleted"):
        await configurations.recover("chat-1", "local")
    assert evidence_calls == []
