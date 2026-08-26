from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


class SandboxFiles:
    def __init__(self, root: Path, *, max_file_bytes: int, max_chat_bytes: int) -> None:
        self._root = root.resolve()
        self._max_file_bytes = max_file_bytes
        self._max_chat_bytes = max_chat_bytes
        self._locks: dict[str, asyncio.Lock] = {}

    async def _mutate(self, chat_id: str, operation: Callable[[], T]) -> T:
        """Serialize a Chat mutation and let blocking filesystem work finish."""

        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            task = asyncio.create_task(asyncio.to_thread(operation))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await asyncio.gather(task, return_exceptions=True)
                raise

    def _chat_root(self, chat_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", chat_id):
            raise ValueError("Invalid Chat identifier")
        return self._root / chat_id

    def _owned_dir(self, chat_id: str, name: str) -> Path:
        chat_root = self._chat_root(chat_id)
        if chat_root.is_symlink():
            raise RuntimeError("Chat directory must not be a symbolic link")
        chat_root.mkdir(parents=True, exist_ok=True)
        path = chat_root / name
        if path.is_symlink():
            raise RuntimeError(f"Chat {name} directory must not be a symbolic link")
        path.mkdir(exist_ok=True)
        if not path.resolve().is_relative_to(chat_root.resolve()):
            raise RuntimeError(f"Chat {name} directory escaped its root")
        return path

    def files_dir(self, chat_id: str) -> Path:
        return self._owned_dir(chat_id, "files")

    def environment_dir(self, chat_id: str) -> Path:
        """Return runtime state kept outside revisioned Chat files."""

        return self._owned_dir(chat_id, "environment")

    def artifacts_dir(self, chat_id: str) -> Path:
        """Return revisioned internal files used to offload model context."""

        return self._owned_dir(chat_id, "artifacts")

    def manifest(self, chat_id: str) -> dict[str, str]:
        root = self.files_dir(chat_id)
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        }

    @staticmethod
    def _unique_destination(destination: Path) -> Path:
        candidate = destination
        index = 2
        while candidate.exists() or candidate.is_symlink():
            candidate = destination.with_name(
                f"{destination.stem} ({index}){destination.suffix}"
            )
            index += 1
        return candidate

    async def upload(
        self, chat_id: str, source: Path, name: str, *, replace: bool = False
    ) -> Path:
        """Store an Uploaded File without silently replacing a name collision."""

        return await self._mutate(
            chat_id,
            lambda: self._upload(chat_id, source, name, replace=replace),
        )

    def _upload(self, chat_id: str, source: Path, name: str, *, replace: bool) -> Path:
        size = source.stat().st_size
        if size > self._max_file_bytes:
            raise ValueError("Uploaded File exceeds the file limit")
        files = self.files_dir(chat_id)
        requested_destination = files / Path(name).name
        destination = (
            requested_destination
            if replace
            else self._unique_destination(requested_destination)
        )
        current_size = (
            sum(item.stat().st_size for item in files.rglob("*") if item.is_file())
            if files.exists()
            else 0
        )
        replaced_size = destination.stat().st_size if destination.is_file() else 0
        if current_size - replaced_size + size > self._max_chat_bytes:
            raise ValueError("Uploaded File exceeds the Chat limit")
        staging = self._owned_dir(chat_id, "staging")
        temporary = staging / f"upload-{uuid.uuid4().hex}"
        try:
            shutil.copy2(source, temporary)
            copied_size = temporary.stat().st_size
            if copied_size > self._max_file_bytes:
                raise ValueError("Uploaded File exceeds the file limit")
            if current_size - replaced_size + copied_size > self._max_chat_bytes:
                raise ValueError("Uploaded File exceeds the Chat limit")
            temporary.replace(destination)
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
        return destination

    async def snapshot(self, chat_id: str) -> str:
        return await self._mutate(chat_id, lambda: self._snapshot(chat_id))

    def _snapshot(self, chat_id: str) -> str:
        chat_root = self._chat_root(chat_id)
        snapshot_id = uuid.uuid4().hex
        snapshot = self._owned_dir(chat_id, "snapshots") / snapshot_id
        snapshot.mkdir()
        try:
            for name in ("files", "artifacts"):
                source = chat_root / name
                destination = snapshot / name
                if source.exists():
                    shutil.copytree(source, destination, symlinks=True)
                else:
                    destination.mkdir()
            (snapshot / "snapshot.json").write_text(
                json.dumps({"version": 2}, separators=(",", ":")),
                encoding="utf-8",
            )
        except BaseException:
            shutil.rmtree(snapshot, ignore_errors=True)
            raise
        return snapshot_id

    async def restore(self, chat_id: str, snapshot: str) -> None:
        await self._mutate(chat_id, lambda: self._restore(chat_id, snapshot))

    @staticmethod
    def _path_exists(path: Path) -> bool:
        return path.exists() or path.is_symlink()

    @classmethod
    def _remove_path(cls, path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)

    def _restore(self, chat_id: str, snapshot: str) -> None:
        if not re.fullmatch(r"[a-f0-9]{32}", snapshot):
            raise ValueError("Invalid Sandbox snapshot")
        chat_root = self._chat_root(chat_id)
        source = chat_root / "snapshots" / snapshot
        if not source.is_dir():
            raise KeyError(snapshot)
        metadata = source / "snapshot.json"
        if metadata.is_file():
            try:
                version = json.loads(metadata.read_text(encoding="utf-8"))["version"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError("Invalid Sandbox snapshot metadata") from error
            if version != 2:
                raise ValueError(f"Unsupported Sandbox snapshot version: {version}")
            sources = {name: source / name for name in ("files", "artifacts")}
            if not all(path.is_dir() for path in sources.values()):
                raise ValueError("Invalid Sandbox snapshot contents")
        else:
            # Version 1 stored the user file tree directly at the snapshot root
            # and predates revisioned internal artifacts.
            sources = {"files": source, "artifacts": None}

        staging = self._owned_dir(chat_id, "staging") / f"restore-{uuid.uuid4().hex}"
        revisions = staging / "revisions"
        backups = staging / "backups"
        revisions.mkdir(parents=True)
        backups.mkdir()
        names = tuple(sources)
        installed: set[str] = set()
        try:
            for name, revision in sources.items():
                staged = revisions / name
                if revision is None:
                    staged.mkdir()
                else:
                    shutil.copytree(revision, staged, symlinks=True)

            for name in names:
                destination = chat_root / name
                backup = backups / name
                if self._path_exists(destination):
                    destination.replace(backup)
                (revisions / name).replace(destination)
                installed.add(name)
        except BaseException:
            for name in reversed(names):
                destination = chat_root / name
                backup = backups / name
                if name in installed:
                    self._remove_path(destination)
                if self._path_exists(backup):
                    backup.replace(destination)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    async def delete_chat(self, chat_id: str) -> None:
        await self._mutate(chat_id, lambda: self._delete_chat(chat_id))

    def _delete_chat(self, chat_id: str) -> None:
        chat_root = self._chat_root(chat_id)
        if chat_root.exists():
            shutil.rmtree(chat_root)
