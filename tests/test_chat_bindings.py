import sqlite3
from dataclasses import FrozenInstanceError

import pytest

from local_agent_chat.chat_bindings import ChatBindings


def test_profile_selection_is_persisted_and_immutable(tmp_path):
    path = tmp_path / "state.sqlite3"
    bindings = ChatBindings(path, ["one", "two"])
    chosen = bindings.open("chat", "two")
    assert chosen.profile_id == "two"
    with pytest.raises(FrozenInstanceError):
        chosen.profile_id = "one"
    assert bindings.open("chat", "one").profile_id == "two"
    assert ChatBindings(path, ["one", "two"]).get("chat") == chosen


def test_removed_profile_falls_back_and_deletion_blocks_reopening(tmp_path):
    path = tmp_path / "state.sqlite3"
    ChatBindings(path, ["old"]).open("chat", "old")
    current = ChatBindings(path, ["one", "two"])
    assert current.open("chat", "old", "two").profile_id == "two"
    assert current.open("new", "missing").profile_id == "one"
    current.delete("chat")
    assert current.get("chat") is None
    with pytest.raises(RuntimeError, match="deleted"):
        current.open("chat")


@pytest.mark.parametrize("table", ["chat_bindings", "active_branches"])
@pytest.mark.parametrize("mode", ["extended", "read_only", "host_files", "chat_files"])
def test_legacy_bindings_keep_profile_and_remove_capability_modes(
    tmp_path, table, mode
):
    path = tmp_path / "state.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute(
            f"CREATE TABLE {table} (chat_id TEXT, profile_id TEXT, agent_mode TEXT)"
        )
        db.execute(f"INSERT INTO {table} VALUES (?, ?, ?)", ("chat", "two", mode))
    bindings = ChatBindings(path, ["one", "two"])
    assert bindings.get("chat").profile_id == "two"
    assert not hasattr(bindings.get("chat"), "mode")
    with sqlite3.connect(path) as db:
        assert not db.execute(
            "SELECT name FROM sqlite_master WHERE name=?", (table,)
        ).fetchall()
    assert ChatBindings(path, ["one", "two"]).get("chat").profile_id == "two"
