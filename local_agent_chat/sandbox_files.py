from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path


class SandboxFiles:
    def __init__(self, root: Path, *, max_file_bytes: int, max_chat_bytes: int) -> None:
        self._root = root.resolve()
        self._max_file_bytes = max_file_bytes
        self._max_chat_bytes = max_chat_bytes

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

    async def upload(self, chat_id: str, source: Path, name: str) -> Path:
        size = source.stat().st_size
        if size > self._max_file_bytes:
            raise ValueError("Uploaded File exceeds the file limit")
        files = self.files_dir(chat_id)
        current_size = (
            sum(item.stat().st_size for item in files.rglob("*") if item.is_file())
            if files.exists()
            else 0
        )
        if current_size + size > self._max_chat_bytes:
            raise ValueError("Uploaded File exceeds the Chat limit")
        destination = files / Path(name).name
        shutil.copy2(source, destination)
        return destination

    async def snapshot(self, chat_id: str) -> str:
        chat_root = self._chat_root(chat_id)
        snapshot_id = uuid.uuid4().hex
        snapshot = chat_root / "snapshots" / snapshot_id
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.mkdir()
        for name in ("files", "artifacts"):
            source = chat_root / name
            destination = snapshot / name
            if source.exists():
                shutil.copytree(source, destination)
            else:
                destination.mkdir()
        (snapshot / "snapshot.json").write_text(
            json.dumps({"version": 2}, separators=(",", ":")),
            encoding="utf-8",
        )
        return snapshot_id

    async def restore(self, chat_id: str, snapshot: str) -> None:
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

        for name, revision in sources.items():
            destination = chat_root / name
            if destination.exists():
                shutil.rmtree(destination)
            if revision is None:
                destination.mkdir()
            else:
                shutil.copytree(revision, destination)

    async def delete_chat(self, chat_id: str) -> None:
        chat_root = self._chat_root(chat_id)
        if chat_root.exists():
            shutil.rmtree(chat_root)
