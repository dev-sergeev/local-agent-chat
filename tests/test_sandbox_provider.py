from pathlib import Path

import pytest

from local_agent_chat.sandbox_files import SandboxFiles
from local_agent_chat.sandbox_provider import LocalSandboxManager


@pytest.mark.asyncio
async def test_local_sandbox_runs_python_in_chat_files_without_app_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-sandbox")
    files = SandboxFiles(
        tmp_path / "sandboxes", max_file_bytes=100, max_chat_bytes=1000
    )
    files.files_dir("chat-1").joinpath("input.txt").write_text(
        "hello", encoding="utf-8"
    )
    manager = LocalSandboxManager(files)

    backend = await manager.backend("chat-1")
    result = await backend.aexecute(
        "python -c 'from pathlib import Path; "
        'Path("result.txt").write_text(Path("input.txt").read_text().upper())\''
    )
    environment = await backend.aexecute("env")

    assert result.exit_code == 0, result.output
    assert (
        files.files_dir("chat-1").joinpath("result.txt").read_text(encoding="utf-8")
        == "HELLO"
    )
    assert "OPENAI_API_KEY" not in environment.output
    assert await manager.backend("chat-1") is backend
