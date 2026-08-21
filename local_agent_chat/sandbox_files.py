from __future__ import annotations

import hashlib
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

    def files_dir(self, chat_id: str) -> Path:
        path = self._chat_root(chat_id) / "files"
        path.mkdir(parents=True, exist_ok=True)
        return path

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
        files = self._chat_root(chat_id) / "files"
        current_size = (
            sum(item.stat().st_size for item in files.rglob("*") if item.is_file())
            if files.exists()
            else 0
        )
        if current_size + size > self._max_chat_bytes:
            raise ValueError("Uploaded File exceeds the Chat limit")
        files.mkdir(parents=True, exist_ok=True)
        destination = files / Path(name).name
        shutil.copy2(source, destination)
        return destination

    async def snapshot(self, chat_id: str) -> str:
        chat_root = self._chat_root(chat_id)
        files = chat_root / "files"
        snapshot_id = uuid.uuid4().hex
        snapshot = chat_root / "snapshots" / snapshot_id
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if files.exists():
            shutil.copytree(files, snapshot)
        else:
            snapshot.mkdir()
        return snapshot_id

    async def restore(self, chat_id: str, snapshot: str) -> None:
        if not re.fullmatch(r"[a-f0-9]{32}", snapshot):
            raise ValueError("Invalid Sandbox snapshot")
        chat_root = self._chat_root(chat_id)
        source = chat_root / "snapshots" / snapshot
        if not source.is_dir():
            raise KeyError(snapshot)
        files = chat_root / "files"
        if files.exists():
            shutil.rmtree(files)
        shutil.copytree(source, files)

    async def delete_chat(self, chat_id: str) -> None:
        chat_root = self._chat_root(chat_id)
        if chat_root.exists():
            shutil.rmtree(chat_root)
