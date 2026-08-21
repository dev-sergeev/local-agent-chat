from __future__ import annotations

import mimetypes
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from chainlit.data.storage_clients.base import BaseStorageClient


class LocalStorageClient(BaseStorageClient):
    def __init__(self, root: Path, public_prefix: str = "/files") -> None:
        self.root = root.resolve()
        self.public_prefix = public_prefix.rstrip("/")

    def path_for(self, object_key: str) -> Path:
        key = PurePosixPath(object_key)
        if key.is_absolute() or ".." in key.parts:
            raise ValueError("Invalid object key")
        path = (self.root / Path(*key.parts)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Invalid object key")
        return path

    async def upload_file(
        self,
        object_key: str,
        data: bytes | str,
        mime: str = "application/octet-stream",
        overwrite: bool = True,
        content_disposition: str | None = None,
    ) -> dict:
        path = self.path_for(object_key)
        if path.exists() and not overwrite:
            raise FileExistsError(object_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode() if isinstance(data, str) else data)
        return {"object_key": object_key, "url": await self.get_read_url(object_key)}

    async def delete_file(self, object_key: str) -> bool:
        path = self.path_for(object_key)
        if not path.exists():
            return False
        path.unlink()
        return True

    async def get_read_url(self, object_key: str) -> str:
        return f"{self.public_prefix}/{quote(object_key, safe='/')}"

    async def close(self) -> None:
        return None

    def media_type(self, object_key: str) -> str:
        return (
            mimetypes.guess_type(self.path_for(object_key).name)[0]
            or "application/octet-stream"
        )
