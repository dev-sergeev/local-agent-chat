from pathlib import Path

import pytest

from local_agent_chat.agent_service import AgentService
from local_agent_chat.settings import ModelProfile


class UnusedSandboxes:
    async def backend(self, chat_id: str):
        raise AssertionError("backend should not be created while selecting a profile")


def profile() -> ModelProfile:
    return ModelProfile("local", "Local", "openai:test", "TEST_KEY", "key")


def test_model_profile_is_immutable_and_survives_restart(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    service = AgentService(database, (profile(),), UnusedSandboxes())  # type: ignore[arg-type]
    service.set_profile("chat-1", "local")

    reopened = AgentService(database, (profile(),), UnusedSandboxes())  # type: ignore[arg-type]
    reopened.set_profile("chat-1", "local")
    with pytest.raises(ValueError, match="Unknown Model Profile"):
        reopened.set_profile("chat-1", "missing")


@pytest.mark.asyncio
async def test_deleting_chat_removes_persisted_model_profile(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    service = AgentService(database, (profile(),), UnusedSandboxes())  # type: ignore[arg-type]
    service.set_profile("chat-1", "local")
    await service.delete_chat("chat-1")
    await service.close()

    reopened = AgentService(database, (profile(),), UnusedSandboxes())  # type: ignore[arg-type]
    assert reopened.profile_for("chat-1") is None
