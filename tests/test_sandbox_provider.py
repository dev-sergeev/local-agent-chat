import asyncio
import json
import os
import sys
import threading
from pathlib import Path

import pytest
from deepagents.middleware.filesystem import supports_execution

from local_agent_chat.agent_modes import (
    EXTENDED_FILESYSTEM_TOOLS,
    READ_ONLY_FILESYSTEM_TOOLS,
    AgentMode,
)
from local_agent_chat.sandbox_files import SandboxFiles
from local_agent_chat.sandbox_provider import LocalSandboxManager


def test_agent_modes_have_explicit_monotonic_tool_allowlists() -> None:
    assert READ_ONLY_FILESYSTEM_TOOLS == ("ls", "read_file", "glob", "grep")
    assert EXTENDED_FILESYSTEM_TOOLS == (
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
        "glob",
        "grep",
        "execute",
    )
    assert set(READ_ONLY_FILESYSTEM_TOOLS) < set(EXTENDED_FILESYSTEM_TOOLS)


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
    victim = tmp_path / "must-not-be-cleared"
    victim.mkdir()
    victim.joinpath("sentinel").write_text("safe", encoding="utf-8")
    environment_root = files.environment_dir("chat-1")
    environment_root.joinpath("venv").symlink_to(victim, target_is_directory=True)
    manager = LocalSandboxManager(files)

    backend = await manager.backend("chat-1", AgentMode.EXTENDED)
    result = await backend.aexecute(
        "python -c 'from pathlib import Path; "
        'Path("result.txt").write_text(Path("input.txt").read_text().upper())\''
    )
    environment = await backend.aexecute(
        "python -c 'import importlib.util,json,os,sys; "
        'print(json.dumps({"executable":sys.executable,"prefix":sys.prefix,'
        '"home":os.environ["HOME"],"tmp":os.environ["TMPDIR"],'
        '"virtual_env":os.environ["VIRTUAL_ENV"],"path":os.environ["PATH"],'
        '"deepagents":importlib.util.find_spec("deepagents") is not None}))\''
    )
    pip = await backend.aexecute("python -m pip --version")

    assert result.exit_code == 0, result.output
    assert environment.exit_code == 0, environment.output
    details = json.loads(environment.output)
    virtualenv = environment_root / "venv"
    executable_dir = virtualenv / ("Scripts" if os.name == "nt" else "bin")
    assert (
        files.files_dir("chat-1").joinpath("result.txt").read_text(encoding="utf-8")
        == "HELLO"
    )
    assert Path(details["prefix"]) == virtualenv
    assert Path(details["executable"]).is_relative_to(virtualenv)
    assert Path(details["home"]).is_relative_to(environment_root)
    assert Path(details["tmp"]).is_relative_to(environment_root)
    assert details["virtual_env"] == str(virtualenv)
    assert details["path"].split(os.pathsep)[0] == str(executable_dir)
    if sys.prefix != sys.base_prefix:
        assert str(Path(sys.executable).parent) not in details["path"].split(os.pathsep)
    assert details["deepagents"] is False
    assert str(virtualenv) in pip.output
    assert "OPENAI_API_KEY" not in (await backend.aexecute("env")).output
    assert set(files.manifest("chat-1")) == {"input.txt", "result.txt"}
    assert victim.joinpath("sentinel").read_text(encoding="utf-8") == "safe"
    assert virtualenv.joinpath(".local-agent-chat-ready").is_file()
    assert await manager.backend("chat-1", AgentMode.EXTENDED) is backend


@pytest.mark.asyncio
async def test_file_tools_and_shell_share_absolute_chat_paths(
    tmp_path: Path,
) -> None:
    files = SandboxFiles(
        tmp_path / "sandboxes", max_file_bytes=100, max_chat_bytes=1000
    )
    backend = await LocalSandboxManager(files).backend("chat-1", AgentMode.EXTENDED)
    script = files.files_dir("chat-1") / "test_things.py"

    write = await backend.awrite(str(script), 'print("reachable")\n')
    result = await backend.aexecute("python test_things.py")

    assert write.error is None
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "reachable"
    assert script.is_file()
    mirrored = files.files_dir("chat-1") / str(script).lstrip("/")
    assert not mirrored.exists()


@pytest.mark.asyncio
async def test_internal_artifact_paths_route_to_revisioned_chat_storage(
    tmp_path: Path,
) -> None:
    files = SandboxFiles(
        tmp_path / "sandboxes", max_file_bytes=100, max_chat_bytes=1000
    )
    backend = await LocalSandboxManager(files).backend("chat-1", AgentMode.READ_ONLY)
    artifacts = files.artifacts_dir("chat-1")
    artifact = artifacts / "large_tool_results" / "result.txt"

    write = await backend.awrite(str(artifact), "stored in the Chat")

    assert write.error is None
    assert artifact.read_text(encoding="utf-8") == "stored in the Chat"
    assert backend.artifacts_root == str(artifacts)
    assert set(backend.routes) == {f"{artifacts}/"}
    assert files.manifest("chat-1") == {}


@pytest.mark.asyncio
async def test_read_only_backend_reads_absolute_host_paths_without_mutation_or_venv(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external.txt"
    external.write_text("host content", encoding="utf-8")
    files = SandboxFiles(
        tmp_path / "sandboxes", max_file_bytes=100, max_chat_bytes=1000
    )
    backend = await LocalSandboxManager(files).backend("chat-1", AgentMode.READ_ONLY)

    read = await backend.aread(str(external))
    write = await backend.awrite(str(external), "changed")
    edit = await backend.aedit(str(external), "host", "guest")
    delete = await backend.adelete(str(external))

    assert read.error is None
    assert read.file_data is not None
    assert read.file_data["content"] == "host content"
    assert write.error and "Read-only" in write.error
    assert edit.error and "Read-only" in edit.error
    assert delete.error and "Read-only" in delete.error
    assert external.read_text(encoding="utf-8") == "host content"
    assert supports_execution(backend) is False
    with pytest.raises(NotImplementedError, match="doesn't support"):
        await backend.aexecute("true")
    assert not (tmp_path / "sandboxes" / "chat-1" / "environment").exists()


@pytest.mark.asyncio
async def test_extended_file_tools_and_shell_share_global_absolute_paths(
    tmp_path: Path,
) -> None:
    files = SandboxFiles(
        tmp_path / "sandboxes", max_file_bytes=100, max_chat_bytes=1000
    )
    backend = await LocalSandboxManager(files).backend("chat-1", AgentMode.EXTENDED)
    external = tmp_path / "outside-chat.txt"

    write = await backend.awrite(str(external), "same path")
    command = await backend.aexecute(
        "python -c 'from pathlib import Path; "
        f"print(Path({json.dumps(str(external))}).read_text())'"
    )

    assert write.error is None
    assert command.exit_code == 0, command.output
    assert command.output.strip() == "same path"
    assert external.read_text(encoding="utf-8") == "same path"
    assert supports_execution(backend) is True


@pytest.mark.asyncio
async def test_each_chat_gets_a_distinct_persistent_python_environment(
    tmp_path: Path,
) -> None:
    files = SandboxFiles(
        tmp_path / "sandboxes", max_file_bytes=100, max_chat_bytes=1000
    )
    first_manager = LocalSandboxManager(files)
    first = await first_manager.backend("chat-1", AgentMode.EXTENDED)
    second = await first_manager.backend("chat-2", AgentMode.EXTENDED)

    first_prefix = await first.aexecute("python -c 'import sys;print(sys.prefix)'")
    second_prefix = await second.aexecute("python -c 'import sys;print(sys.prefix)'")

    assert first_prefix.output.strip() != second_prefix.output.strip()
    reopened = await LocalSandboxManager(files).backend("chat-1", AgentMode.EXTENDED)
    reopened_prefix = await reopened.aexecute(
        "python -c 'import sys;print(sys.prefix)'"
    )
    assert reopened_prefix.output.strip() == first_prefix.output.strip()


@pytest.mark.asyncio
async def test_cancelled_backend_bootstrap_is_shared_with_the_next_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = SandboxFiles(
        tmp_path / "sandboxes", max_file_bytes=100, max_chat_bytes=1000
    )
    manager = LocalSandboxManager(files)
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def slow_environment(_chat_id: str) -> dict[str, str]:
        nonlocal calls
        calls += 1
        started.set()
        release.wait(timeout=5)
        return {"PATH": os.defpath}

    monkeypatch.setattr(manager, "_command_environment", slow_environment)
    first = asyncio.create_task(manager.backend("chat-1", AgentMode.EXTENDED))
    assert await asyncio.to_thread(started.wait, 2)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    second = asyncio.create_task(manager.backend("chat-1", AgentMode.EXTENDED))
    await asyncio.sleep(0)
    release.set()

    assert await second is not None
    assert calls == 1


@pytest.mark.asyncio
async def test_delete_waits_for_bootstrap_and_prevents_chat_recreation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = SandboxFiles(
        tmp_path / "sandboxes", max_file_bytes=100, max_chat_bytes=1000
    )
    manager = LocalSandboxManager(files)
    started = threading.Event()
    release = threading.Event()

    def slow_environment(chat_id: str) -> dict[str, str]:
        started.set()
        release.wait(timeout=5)
        files.environment_dir(chat_id).joinpath("created").write_text("done")
        return {"PATH": os.defpath}

    monkeypatch.setattr(manager, "_command_environment", slow_environment)
    backend_task = asyncio.create_task(manager.backend("chat-1", AgentMode.EXTENDED))
    assert await asyncio.to_thread(started.wait, 2)
    delete_task = asyncio.create_task(manager.delete_chat("chat-1"))
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(RuntimeError, match="being deleted"):
        await backend_task
    await delete_task
    await files.delete_chat("chat-1")
    assert not (tmp_path / "sandboxes" / "chat-1").exists()
    with pytest.raises(RuntimeError, match="being deleted"):
        await manager.backend("chat-1", AgentMode.EXTENDED)
    assert not (tmp_path / "sandboxes" / "chat-1").exists()


@pytest.mark.asyncio
async def test_manager_keeps_chat_mode_immutable(tmp_path: Path) -> None:
    files = SandboxFiles(
        tmp_path / "sandboxes", max_file_bytes=100, max_chat_bytes=1000
    )
    manager = LocalSandboxManager(files)

    backend = await manager.backend("chat-1", AgentMode.READ_ONLY)

    assert await manager.backend("chat-1", AgentMode.READ_ONLY) is backend
    with pytest.raises(ValueError, match="immutable"):
        await manager.backend("chat-1", AgentMode.EXTENDED)
