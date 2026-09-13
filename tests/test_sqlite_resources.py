import gc
import os
from pathlib import Path

import pytest

from local_agent_chat.agent_memory import AgentMemory
from local_agent_chat.chat_bindings import ChatBindings
from local_agent_chat.runtime import Turn
from local_agent_chat.sqlite_history import SQLiteHistory


@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="Linux fd accounting")
@pytest.mark.parametrize("store", ["context", "profiles", "history"])
async def test_repeated_sqlite_operations_close_files_without_garbage_collection(
    tmp_path, store
):
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        path = tmp_path / "state.sqlite3"
        if store == "context":
            memory = AgentMemory(path)
            for _ in range(100):
                memory.save("chat", [])
                assert memory.load("chat") == []
        elif store == "profiles":
            bindings = ChatBindings(path, ["test"])
            for _ in range(100):
                bindings.open("chat")
                assert bindings.get("chat").profile_id == "test"
        else:
            history = SQLiteHistory(path)
            for i in range(100):
                turn = Turn(
                    f"t-{i}", "chat", "request", "answer", "checkpoint", "snapshot"
                )
                await history.append(turn)
                assert (await history.get(turn.id)).text == "request"
        open_files = []
        for fd in Path("/proc/self/fd").iterdir():
            try:
                open_files.append(os.readlink(fd))
            except FileNotFoundError:
                pass
        assert not [name for name in open_files if str(tmp_path) in name]
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()
