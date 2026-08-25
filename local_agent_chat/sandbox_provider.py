from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

from deepagents.backends import (
    BackendProtocol,
    CompositeBackend,
    FilesystemBackend,
    LocalShellBackend,
)
from deepagents.backends.protocol import (
    DeleteResult,
    EditResult,
    FileUploadResponse,
    WriteResult,
)

from .agent_modes import AgentMode
from .sandbox_files import SandboxFiles

_READ_ONLY_ERROR = "Read-only Agent Mode does not allow filesystem changes"


class _ReadOnlyHostBackend(BackendProtocol):
    """Expose host reads while denying every mutation and command execution."""

    def __init__(self) -> None:
        self._filesystem = FilesystemBackend(root_dir="/", virtual_mode=False)

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


def _artifact_route(path: Path) -> str:
    return f"{path.resolve().as_posix().rstrip('/')}/"


class LocalSandboxManager:
    """Keep one local command backend in memory for each Chat."""

    def __init__(self, files: SandboxFiles) -> None:
        self._files = files
        self._backends: dict[str, CompositeBackend] = {}
        self._modes: dict[str, AgentMode] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._environment_tasks: dict[str, asyncio.Task[dict[str, str]]] = {}
        self._deleting: set[str] = set()

    def files_dir(self, chat_id: str) -> Path:
        """Return the absolute working directory used by Extended mode."""

        return self._files.files_dir(chat_id)

    @staticmethod
    def _remove_environment(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)

    @staticmethod
    def _environment_is_ready(virtualenv: Path, *, require_marker: bool) -> bool:
        executable_dir = virtualenv / ("Scripts" if os.name == "nt" else "bin")
        python = executable_dir / ("python.exe" if os.name == "nt" else "python")
        marker = virtualenv / ".local-agent-chat-ready"
        expected = f"{sys.version_info.major}.{sys.version_info.minor}"
        if virtualenv.is_symlink() or not virtualenv.is_dir():
            return False
        if require_marker:
            if marker.is_symlink() or not marker.is_file():
                return False
            if marker.read_text(encoding="utf-8").strip() != expected:
                return False
        if (
            python.is_symlink()
            or not python.is_file()
            or not python.resolve().is_relative_to(virtualenv.resolve())
            or not (virtualenv / "pyvenv.cfg").is_file()
        ):
            return False
        try:
            check = subprocess.run(
                [str(python), "-m", "pip", "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
                env={"LANG": "C.UTF-8", "PATH": os.defpath},
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return check.returncode == 0 and str(virtualenv) in check.stdout

    def _command_environment(self, chat_id: str) -> dict[str, str]:
        root = self._files.environment_dir(chat_id)
        virtualenv = root / "venv"
        executable_dir = virtualenv / ("Scripts" if os.name == "nt" else "bin")
        if not self._environment_is_ready(virtualenv, require_marker=True):
            self._remove_environment(virtualenv)
            venv.EnvBuilder(
                with_pip=True,
                clear=False,
                symlinks=False,
            ).create(virtualenv)
            if not self._environment_is_ready(virtualenv, require_marker=False):
                raise RuntimeError("Failed to create the Chat Python environment")
            (virtualenv / ".local-agent-chat-ready").write_text(
                f"{sys.version_info.major}.{sys.version_info.minor}\n",
                encoding="utf-8",
            )

        home = root / "home"
        temporary = root / "tmp"
        cache = root / "cache"
        for directory in (home, temporary, cache):
            directory.mkdir(exist_ok=True)

        return {
            "HOME": str(home),
            "LANG": "C.UTF-8",
            "PATH": os.pathsep.join((str(executable_dir), os.defpath)),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_REQUIRE_VIRTUALENV": "1",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": str(temporary),
            "VIRTUAL_ENV": str(virtualenv),
            "XDG_CACHE_HOME": str(cache),
        }

    def _environment_finished(
        self, chat_id: str, task: asyncio.Task[dict[str, str]]
    ) -> None:
        if self._environment_tasks.get(chat_id) is task:
            self._environment_tasks.pop(chat_id, None)
        if not task.cancelled():
            task.exception()

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
            if chat_id not in self._backends and mode is AgentMode.EXTENDED:
                task = self._environment_tasks.get(chat_id)
                if task is None:
                    task = asyncio.create_task(
                        asyncio.to_thread(self._command_environment, chat_id)
                    )
                    self._environment_tasks[chat_id] = task
                    task.add_done_callback(
                        lambda finished: self._environment_finished(chat_id, finished)
                    )
                environment = await asyncio.shield(task)
                if chat_id in self._deleting:
                    raise RuntimeError("Chat is being deleted")
                default: BackendProtocol = LocalShellBackend(
                    root_dir=self._files.files_dir(chat_id),
                    virtual_mode=False,
                    env=environment,
                    inherit_env=False,
                )
            elif chat_id not in self._backends:
                default = _ReadOnlyHostBackend()

            if chat_id not in self._backends:
                artifacts = self._files.artifacts_dir(chat_id).resolve()
                self._backends[chat_id] = CompositeBackend(
                    default=default,
                    routes={
                        _artifact_route(artifacts): FilesystemBackend(
                            root_dir=artifacts,
                            virtual_mode=True,
                        )
                    },
                    artifacts_root=artifacts.as_posix(),
                )
                self._modes[chat_id] = mode
            return self._backends[chat_id]

    async def push(self, chat_id: str, backend: CompositeBackend) -> None:
        """Files already live in the backend's Chat directory."""

    async def pull(self, chat_id: str, backend: CompositeBackend) -> None:
        """Command results are written directly into the Chat directory."""

    async def delete_chat(self, chat_id: str) -> None:
        self._deleting.add(chat_id)
        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            task = self._environment_tasks.get(chat_id)
            if task is not None:
                try:
                    await asyncio.shield(task)
                except Exception:  # noqa: BLE001 - deletion still owns partial state
                    pass
            self._backends.pop(chat_id, None)
            self._modes.pop(chat_id, None)
