from __future__ import annotations

import asyncio
from pathlib import Path

from deepagents.backends import BackendProtocol, CompositeBackend, FilesystemBackend
from deepagents.backends.protocol import (
    DeleteResult,
    EditResult,
    FileUploadResponse,
    WriteResult,
)

from .agent_modes import AgentMode
from .sandbox_files import SandboxFiles

_READ_ONLY_ERROR = "Agent file access is read-only"


class _ReadOnlyFilesystemBackend(BackendProtocol):
    """Expose one filesystem scope without mutation or execution."""

    def __init__(self, root_dir: str | Path, *, virtual_mode: bool) -> None:
        self._filesystem = FilesystemBackend(
            root_dir=root_dir,
            virtual_mode=virtual_mode,
        )

    def ls(self, path: str):
        return self._filesystem.ls(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000):
        return self._filesystem.read(file_path, offset=offset, limit=limit)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ):
        return self._filesystem.grep(pattern, path=path, glob=glob, max_count=max_count)

    def glob(self, pattern: str, path: str | None = None):
        return self._filesystem.glob(pattern, path=path)

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error=_READ_ONLY_ERROR)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        return EditResult(error=_READ_ONLY_ERROR)

    def delete(self, file_path: str) -> DeleteResult:
        return DeleteResult(error=_READ_ONLY_ERROR)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [
            FileUploadResponse(path=path, error=_READ_ONLY_ERROR) for path, _ in files
        ]

    def download_files(self, paths: list[str]):
        return self._filesystem.download_files(paths)


def _route(path: Path) -> str:
    return f"{path.resolve().as_posix().rstrip('/')}/"


class LocalSandboxManager:
    """Build one immutable read-scope backend for each Chat."""

    def __init__(
        self,
        files: SandboxFiles,
        *,
        system_read_roots: tuple[Path, ...] = (),
    ) -> None:
        self._files = files
        roots = tuple(dict.fromkeys(path.resolve() for path in system_read_roots))
        missing = next((path for path in roots if not path.is_dir()), None)
        if missing is not None:
            raise ValueError(f"System read root does not exist: {missing}")
        self._system_read_roots = roots
        self._backends: dict[str, CompositeBackend] = {}
        self._modes: dict[str, AgentMode] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._deleting: set[str] = set()

    def files_dir(self, chat_id: str) -> Path:
        """Return the absolute directory that stores Uploaded Files."""

        return self._files.files_dir(chat_id)

    def _build_backend(self, chat_id: str, mode: AgentMode) -> CompositeBackend:
        artifacts = self._files.artifacts_dir(chat_id).resolve()
        routes: dict[str, BackendProtocol] = {
            _route(artifacts): FilesystemBackend(
                root_dir=artifacts,
                virtual_mode=True,
            )
        }
        if mode is AgentMode.CHAT_FILES:
            default: BackendProtocol = _ReadOnlyFilesystemBackend(
                self._files.files_dir(chat_id),
                virtual_mode=True,
            )
            routes.update(
                {
                    _route(root): _ReadOnlyFilesystemBackend(
                        root,
                        virtual_mode=True,
                    )
                    for root in self._system_read_roots
                }
            )
        else:
            default = _ReadOnlyFilesystemBackend("/", virtual_mode=False)

        return CompositeBackend(
            default=default,
            routes=routes,
            artifacts_root=artifacts.as_posix(),
        )

    async def backend(self, chat_id: str, mode: AgentMode) -> CompositeBackend:
        mode = AgentMode(mode)
        if chat_id in self._deleting:
            raise RuntimeError("Chat is being deleted")
        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            if chat_id in self._deleting:
                raise RuntimeError("Chat is being deleted")
            active_mode = self._modes.get(chat_id)
            if active_mode is not None and active_mode is not mode:
                raise ValueError("Agent Mode is immutable after the Chat is bound")
            if chat_id not in self._backends:
                self._backends[chat_id] = self._build_backend(chat_id, mode)
                self._modes[chat_id] = mode
            return self._backends[chat_id]

    async def delete_chat(self, chat_id: str) -> None:
        self._deleting.add(chat_id)
        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            self._backends.pop(chat_id, None)
            self._modes.pop(chat_id, None)
