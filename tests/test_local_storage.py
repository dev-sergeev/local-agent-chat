from pathlib import Path

import pytest

from local_agent_chat.local_storage import LocalStorageClient


@pytest.mark.asyncio
async def test_blob_content_is_local_and_url_contains_only_object_key(
    tmp_path: Path,
) -> None:
    storage = LocalStorageClient(tmp_path, "/prefix/files")
    result = await storage.upload_file("user/chat/report.txt", b"content", "text/plain")
    assert storage.path_for("user/chat/report.txt").read_bytes() == b"content"
    assert result["url"] == "/prefix/files/user/chat/report.txt"
    with pytest.raises(ValueError):
        storage.path_for("../secret")
