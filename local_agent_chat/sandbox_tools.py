"""Bounded, read-only file tools rooted exclusively in one Chat's uploads."""

from __future__ import annotations

import fnmatch
import os
import stat
from pathlib import Path, PurePosixPath

from langchain_core.tools import BaseTool, tool

MAX_OUTPUT_CHARS = 6000
MAX_READ_BYTES = 256 * 1024
MAX_ENTRIES = 1000


class SandboxReader:
    def __init__(self, root: Path) -> None:
        if root.is_symlink():
            raise ValueError("Sandbox root must not be a symbolic link")
        self.root = root.resolve(strict=True)

    def _parts(self, path: str) -> tuple[str, ...]:
        if "\\" in path or "\x00" in path or ".." in PurePosixPath(path).parts:
            raise ValueError(
                "Use a path inside this Chat's sandbox; traversal is forbidden"
            )
        return PurePosixPath("/" + path.lstrip("/")).parts[1:]

    def _open(self, path: str, *, directory: bool = False) -> int:
        # Open each component relative to an owned directory descriptor. Reject
        # symlinks even if a path changes between validation and opening it.
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        fd = os.open(self.root, flags | os.O_DIRECTORY)
        try:
            parts = self._parts(path)
            for index, part in enumerate(parts):
                is_dir = index < len(parts) - 1 or directory
                next_fd = os.open(
                    part, flags | (os.O_DIRECTORY if is_dir else 0), dir_fd=fd
                )
                os.close(fd)
                fd = next_fd
            return fd
        except BaseException:
            os.close(fd)
            raise

    def read(self, path: str, offset: int, limit: int) -> str:
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError("offset must be >= 0; limit must be between 1 and 200")
        fd = self._open(path)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Only regular text files can be read")
            lines = []
            for index, raw in self._lines(stream):
                if index < offset:
                    continue
                if index >= offset + limit:
                    break
                lines.append(f"{index + 1}: {raw}")
                if len(lines) >= limit or sum(map(len, lines)) >= MAX_OUTPUT_CHARS:
                    break
            more = bool(stream.peek(1))
        result = "\n".join(lines)
        if more:
            result += "\n[More content exists; narrow the read or search for a phrase.]"
        return result or "[No lines in this range]"

    @staticmethod
    def _lines(stream):
        index = 0
        while raw := stream.readline(MAX_READ_BYTES):
            if b"\x00" in raw:
                raise ValueError("Binary file")
            # Extremely long lines are bounded but remain one logical line.
            truncated = not raw.endswith(b"\n") and len(raw) == MAX_READ_BYTES
            text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if truncated:
                while tail := stream.readline(MAX_READ_BYTES):
                    if tail.endswith(b"\n"):
                        break
                text += " [Long line truncated]"
            yield index, text
            index += 1

    def search(self, path: str, phrase: str) -> str:
        fd = self._open(path)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Only regular text files can be searched")
            matches = []
            size = 0
            for index, line in self._lines(stream):
                if phrase in line:
                    match = f"{path}:{index + 1}: {line}"
                    matches.append(match)
                    size += len(match)
                    if size >= MAX_OUTPUT_CHARS:
                        return "\n".join(matches) + "\n[More matches may exist.]"
        return "\n".join(matches)

    def entries(self, path: str = "/", *, recursive: bool = False) -> list[str]:
        result: list[str] = []
        start = "/" + "/".join(self._parts(path))
        pending = [start]
        while pending and len(result) < MAX_ENTRIES:
            parent = pending.pop()
            fd = self._open(parent, directory=True)
            try:
                with os.scandir(fd) as entries:
                    for entry in entries:
                        if entry.is_symlink():
                            continue
                        is_dir = entry.is_dir(follow_symlinks=False)
                        if not is_dir and not entry.is_file(follow_symlinks=False):
                            continue
                        name = parent.rstrip("/") + "/" + entry.name
                        result.append(name + ("/" if is_dir else ""))
                        if recursive and is_dir:
                            pending.append(name)
                        if len(result) >= MAX_ENTRIES:
                            break
            finally:
                os.close(fd)
        return sorted(result)


def build_sandbox_tools(root: Path) -> list[BaseTool]:
    reader = SandboxReader(root)

    def bounded(operation) -> str:
        try:
            result = operation()
        except (OSError, ValueError, UnicodeError):
            return "Cannot read this path. Only text files inside this Chat's sandbox are available."
        if len(result) > MAX_OUTPUT_CHARS:
            return result[:MAX_OUTPUT_CHARS] + "\n[Result limited; narrow your query.]"
        return result

    @tool
    def ls(path: str = "/") -> str:
        """List entries in a sandbox directory. Paths are virtual, relative to uploads."""
        return bounded(lambda: "\n".join(reader.entries(path)) or "[Empty directory]")

    @tool
    def read_file(file_path: str, offset: int = 0, limit: int = 100) -> str:
        """Read UTF-8 sandbox text with line numbers; offset is zero-based, limit <= 200."""
        return bounded(lambda: reader.read(file_path, offset, limit))

    @tool
    def glob(pattern: str, path: str = "/") -> str:
        """Find sandbox paths by a wildcard pattern, for example *.txt or reports/*.md."""
        return bounded(
            lambda: (
                "\n".join(
                    name
                    for name in reader.entries(path, recursive=True)
                    if fnmatch.fnmatchcase(name.lstrip("/"), pattern.lstrip("/"))
                    or fnmatch.fnmatchcase(PurePosixPath(name).name, pattern)
                )
                or "[No matches]"
            )
        )

    @tool
    def grep(pattern: str, path: str = "/") -> str:
        """Search for a literal text phrase in sandbox files; results include paths and lines."""

        def search() -> str:
            if not pattern or len(pattern) > 500:
                raise ValueError("Use a nonempty phrase of at most 500 characters")
            try:
                names = reader.entries(path, recursive=True)
            except NotADirectoryError:
                names = [path]
            matches: list[str] = []
            size = 0
            for name in names:
                if name.endswith("/"):
                    continue
                try:
                    text = reader.search(name, pattern)
                except (OSError, ValueError, UnicodeError):
                    continue
                if text:
                    matches.append(text)
                    size += len(text)
                    if size >= MAX_OUTPUT_CHARS:
                        return "\n".join(matches) + "\n[More matches may exist.]"
            return "\n".join(matches) or "[No matches in the bounded search]"

        return bounded(search)

    return [ls, read_file, glob, grep]
