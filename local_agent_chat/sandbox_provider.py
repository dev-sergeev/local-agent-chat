from __future__ import annotations

import os
import sys
from pathlib import Path

from deepagents.backends import LocalShellBackend

from .sandbox_files import SandboxFiles


class LocalSandboxManager:
    """Keep one local command backend in memory for each Chat."""

    def __init__(self, files: SandboxFiles) -> None:
        self._files = files
        self._backends: dict[str, LocalShellBackend] = {}

    async def backend(self, chat_id: str) -> LocalShellBackend:
        if chat_id not in self._backends:
            root = self._files.files_dir(chat_id)
            temporary = root / ".tmp"
            temporary.mkdir(exist_ok=True)
            executable_dir = str(Path(sys.executable).parent)
            self._backends[chat_id] = LocalShellBackend(
                root_dir=root,
                virtual_mode=True,
                env={
                    "HOME": str(root),
                    "LANG": "C.UTF-8",
                    "PATH": os.pathsep.join((executable_dir, os.defpath)),
                    "TMPDIR": str(temporary),
                },
                inherit_env=False,
            )
        return self._backends[chat_id]

    async def push(self, chat_id: str, backend: LocalShellBackend) -> None:
        """Files already live in the backend's Chat directory."""

    async def pull(self, chat_id: str, backend: LocalShellBackend) -> None:
        """Command results are written directly into the Chat directory."""

    async def delete_chat(self, chat_id: str) -> None:
        self._backends.pop(chat_id, None)
