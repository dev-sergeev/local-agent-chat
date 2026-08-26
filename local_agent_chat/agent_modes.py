from __future__ import annotations

from enum import StrEnum


class AgentMode(StrEnum):
    """A Chat's immutable file-reading scope."""

    CHAT_FILES = "chat_files"
    HOST_FILES = "host_files"


# Both modes expose exactly the same read tools. The mode changes only the
# backend scope; keeping one allowlist prevents a dependency upgrade from
# silently restoring file mutation or command execution.
READ_FILESYSTEM_TOOLS = ("ls", "read_file", "glob", "grep")
