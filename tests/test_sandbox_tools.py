import os

import pytest

from local_agent_chat.sandbox_tools import (
    MAX_OUTPUT_CHARS,
    SandboxReader,
    build_sandbox_tools,
)


@pytest.fixture
def files(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    (root / "note.txt").write_text("first\nneedle evidence\nthird\n", encoding="utf-8")
    (root / "nested").mkdir()
    (root / "nested" / "other.txt").write_text("nested evidence")
    return root


async def test_read_list_glob_and_literal_search(files):
    tools = {t.name: t for t in build_sandbox_tools(files)}
    assert set(tools) == {"ls", "read_file", "glob", "grep"}
    assert "/note.txt" in await tools["ls"].ainvoke({})
    assert (
        await tools["read_file"].ainvoke(
            {"file_path": "/note.txt", "offset": 1, "limit": 1}
        )
        == "2: needle evidence\n[More content exists; narrow the read or search for a phrase.]"
    )
    assert "/nested/other.txt" in await tools["glob"].ainvoke({"pattern": "*.txt"})
    assert "/note.txt:2: needle evidence" in await tools["grep"].ainvoke(
        {"pattern": "needle"}
    )
    assert "No matches" in await tools["grep"].ainvoke({"pattern": ".*"})
    for directory in ("nested", "nested/", "/nested"):
        assert "/nested/other.txt:1: nested evidence" in await tools["grep"].ainvoke(
            {"pattern": "evidence", "path": directory}
        )
    assert "/note.txt:2: needle evidence" in await tools["grep"].ainvoke(
        {"pattern": "needle", "path": "/note.txt"}
    )


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "/../outside.txt",
        "nested/../../outside.txt",
        "/etc/passwd",
        "C:\\outside.txt",
    ],
)
async def test_host_and_traversal_are_unavailable(files, path):
    (files.parent / "outside.txt").write_text("HOST_SENTINEL")
    reader = {t.name: t for t in build_sandbox_tools(files)}["read_file"]
    result = await reader.ainvoke({"file_path": path})
    assert "HOST_SENTINEL" not in result
    assert result.startswith("Cannot read")


async def test_symlink_files_and_directories_cannot_escape(files):
    host = files.parent / "host"
    host.mkdir()
    (host / "secret.txt").write_text("HOST_SENTINEL")
    (files / "link.txt").symlink_to(host / "secret.txt")
    (files / "linkdir").symlink_to(host, target_is_directory=True)
    tools = {t.name: t for t in build_sandbox_tools(files)}
    for path in ["/link.txt", "/linkdir/secret.txt"]:
        assert "Cannot read" in await tools["read_file"].ainvoke({"file_path": path})
    for name, args in [
        ("ls", {}),
        ("glob", {"pattern": "*"}),
        ("grep", {"pattern": "HOST_SENTINEL"}),
    ]:
        result = await tools[name].ainvoke(args)
        assert "HOST_SENTINEL" not in result and "link" not in result


async def test_outputs_and_special_files_are_bounded(files):
    (files / "large.txt").write_text("x" * 1000000)
    (files / "binary").write_bytes(b"\x00\xff")
    os.mkfifo(files / "pipe")
    tools = {t.name: t for t in build_sandbox_tools(files)}
    result = await tools["read_file"].ainvoke({"file_path": "/large.txt"})
    assert len(result) <= MAX_OUTPUT_CHARS + 100
    for path in ["/binary", "/pipe"]:
        assert (await tools["read_file"].ainvoke({"file_path": path})).startswith(
            "Cannot read"
        )


async def test_read_can_page_beyond_first_buffer_and_grep_finds_late_lines(files):
    (files / "long.txt").write_text(
        "padding " * 80 + "\n" + ("ordinary line\n" * 30000) + "LATE_TARGET\n"
    )
    tools = {t.name: t for t in build_sandbox_tools(files)}
    result = await tools["read_file"].ainvoke(
        {"file_path": "/long.txt", "offset": 30001, "limit": 1}
    )
    assert "LATE_TARGET" in result
    result = await tools["grep"].ainvoke(
        {"pattern": "LATE_TARGET", "path": "/long.txt"}
    )
    assert "LATE_TARGET" in result


def test_root_symlink_is_rejected(files):
    link = files.parent / "alias"
    link.symlink_to(files, target_is_directory=True)
    with pytest.raises(ValueError):
        SandboxReader(link)
