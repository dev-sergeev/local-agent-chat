from __future__ import annotations

from enum import StrEnum


class AgentMode(StrEnum):
    """A Chat's immutable filesystem and command capability set."""

    READ_ONLY = "read_only"
    EXTENDED = "extended"


# Keep these allowlists explicit: a Deep Agents upgrade must not silently add a
# newly introduced filesystem capability to either mode.
READ_ONLY_FILESYSTEM_TOOLS = ("ls", "read_file", "glob", "grep")
EXTENDED_FILESYSTEM_TOOLS = (
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "delete",
    "glob",
    "grep",
    "execute",
)
