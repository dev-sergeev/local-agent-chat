from pathlib import Path

import pytest
from deepagents.middleware.filesystem import supports_execution

from local_agent_chat.agent_modes import READ_FILESYSTEM_TOOLS, AgentMode
from local_agent_chat.sandbox_files import SandboxFiles
from local_agent_chat.sandbox_provider import LocalSandboxManager

FORBIDDEN_AGENT_TOOLS = {
    "write_file",
    "edit_file",
    "delete",
    "delete_file",
    "execute",
}


def sandbox_files(tmp_path: Path) -> SandboxFiles:
    return SandboxFiles(
        tmp_path / "sandboxes",
        max_file_bytes=1024,
        max_chat_bytes=4096,
    )


def test_agent_modes_share_one_explicit_read_only_tool_allowlist() -> None:
    assert READ_FILESYSTEM_TOOLS == ("ls", "read_file", "glob", "grep")
    assert FORBIDDEN_AGENT_TOOLS.isdisjoint(READ_FILESYSTEM_TOOLS)


async def assert_backend_has_no_generic_mutation_or_execution(
    backend, target: Path
) -> None:
    write = await backend.awrite(str(target), "changed")
    edit = await backend.aedit(str(target), "original", "changed")
    delete = await backend.adelete(str(target))

    assert write.error and "read-only" in write.error
    assert edit.error and "read-only" in edit.error
    assert delete.error and "read-only" in delete.error
    assert supports_execution(backend) is False
    with pytest.raises(NotImplementedError, match="doesn't support"):
        await backend.aexecute("true")


@pytest.mark.asyncio
async def test_chat_files_backend_reads_only_uploaded_files(tmp_path: Path) -> None:
    files = sandbox_files(tmp_path)
    chat_files = files.files_dir("chat-1")
    uploaded = chat_files / "report.txt"
    uploaded.write_text("uploaded content", encoding="utf-8")
    external = tmp_path / "external.txt"
    external.write_text("host content", encoding="utf-8")
    other_chat = files.files_dir("chat-2") / "private.txt"
    other_chat.write_text("other chat", encoding="utf-8")
    escape = chat_files / "escape.txt"
    escape.symlink_to(external)

    manager = LocalSandboxManager(files)
    backend = await manager.backend("chat-1", AgentMode.CHAT_FILES)

    read = await backend.aread("/report.txt")
    assert read.error is None
    assert read.file_data is not None
    assert read.file_data["content"] == "uploaded content"
    assert (await backend.aread(str(external))).error
    assert (await backend.aread(str(other_chat))).error
    with pytest.raises(ValueError, match="outside root"):
        await backend.aread("/escape.txt")
    with pytest.raises(ValueError, match="traversal"):
        await backend.aread("../external.txt")

    late_upload = chat_files / "later.txt"
    late_upload.write_text("available without rebuilding", encoding="utf-8")
    late_read = await backend.aread("/later.txt")
    assert late_read.file_data is not None
    assert late_read.file_data["content"] == "available without rebuilding"

    await assert_backend_has_no_generic_mutation_or_execution(backend, uploaded)
    assert uploaded.read_text(encoding="utf-8") == "uploaded content"
    assert external.read_text(encoding="utf-8") == "host content"


@pytest.mark.asyncio
async def test_host_files_backend_reads_real_absolute_paths_without_mutation(
    tmp_path: Path,
) -> None:
    files = sandbox_files(tmp_path)
    external = tmp_path / "external.txt"
    external.write_text("host content", encoding="utf-8")
    other_chat = files.files_dir("chat-2") / "private.txt"
    other_chat.write_text("other chat", encoding="utf-8")
    backend = await LocalSandboxManager(files).backend("chat-1", AgentMode.HOST_FILES)

    external_read = await backend.aread(str(external))
    other_chat_read = await backend.aread(str(other_chat))

    assert external_read.file_data is not None
    assert external_read.file_data["content"] == "host content"
    assert other_chat_read.file_data is not None
    assert other_chat_read.file_data["content"] == "other chat"
    await assert_backend_has_no_generic_mutation_or_execution(backend, external)
    assert external.read_text(encoding="utf-8") == "host content"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(AgentMode))
async def test_project_skills_are_readable_in_both_modes(
    tmp_path: Path, mode: AgentMode
) -> None:
    files = sandbox_files(tmp_path)
    skills = tmp_path / "skills"
    skill = skills / "incident-summary"
    reference = skill / "references" / "format.md"
    reference.parent.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Incident summary", encoding="utf-8")
    reference.write_text("timeline format", encoding="utf-8")
    adjacent = tmp_path / "not-a-skill.txt"
    adjacent.write_text("private host file", encoding="utf-8")
    backend = await LocalSandboxManager(
        files,
        system_read_roots=(skills,),
    ).backend("chat-1", mode)

    skill_read = await backend.aread(str(skill / "SKILL.md"))
    reference_read = await backend.aread(str(reference))

    assert skill_read.file_data is not None
    assert skill_read.file_data["content"] == "# Incident summary"
    assert reference_read.file_data is not None
    assert reference_read.file_data["content"] == "timeline format"
    if mode is AgentMode.CHAT_FILES:
        assert (await backend.aread(str(adjacent))).error


def test_missing_system_read_root_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="System read root does not exist"):
        LocalSandboxManager(
            sandbox_files(tmp_path),
            system_read_roots=(tmp_path / "missing-skills",),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(AgentMode))
async def test_internal_artifacts_remain_writable_for_context_offload(
    tmp_path: Path, mode: AgentMode
) -> None:
    files = sandbox_files(tmp_path)
    backend = await LocalSandboxManager(files).backend("chat-1", mode)
    artifacts = files.artifacts_dir("chat-1")
    artifact = artifacts / "large_tool_results" / "result.txt"

    write = await backend.awrite(str(artifact), "internal context")

    assert write.error is None
    assert artifact.read_text(encoding="utf-8") == "internal context"
    assert backend.artifacts_root == str(artifacts)
    assert files.manifest("chat-1") == {}


@pytest.mark.asyncio
async def test_manager_keeps_chat_mode_immutable(tmp_path: Path) -> None:
    manager = LocalSandboxManager(sandbox_files(tmp_path))
    backend = await manager.backend("chat-1", AgentMode.CHAT_FILES)

    assert await manager.backend("chat-1", AgentMode.CHAT_FILES) is backend
    with pytest.raises(ValueError, match="immutable"):
        await manager.backend("chat-1", AgentMode.HOST_FILES)


@pytest.mark.asyncio
async def test_deleted_chat_cannot_recreate_a_backend(tmp_path: Path) -> None:
    manager = LocalSandboxManager(sandbox_files(tmp_path))
    await manager.backend("chat-1", AgentMode.CHAT_FILES)

    await manager.delete_chat("chat-1")

    with pytest.raises(RuntimeError, match="being deleted"):
        await manager.backend("chat-1", AgentMode.CHAT_FILES)
