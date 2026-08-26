from pathlib import Path

import chainlit.server as chainlit_server
import pytest
from chainlit.config import config
from chainlit.session import WebsocketSession

from local_agent_chat.chainlit_uploads import (
    install_empty_file_upload_compatibility,
    install_unrestricted_file_upload_compatibility,
)


@pytest.mark.asyncio
async def test_empty_upload_uses_chainlit_path_contract_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[tuple[str | None, bytes | str | None]] = []

    async def chainlit_persist_file(
        self: WebsocketSession,
        name: str,
        mime: str,
        path: str | None = None,
        content: bytes | str | None = None,
    ) -> dict[str, str]:
        del self, name, mime
        if not path and not content:
            raise ValueError("Either path or content must be provided")
        received.append((path, content))
        assert path is not None
        assert Path(path).read_bytes() == b""
        return {"id": "empty-file"}

    monkeypatch.setattr(WebsocketSession, "persist_file", chainlit_persist_file)
    install_empty_file_upload_compatibility()
    installed_method = WebsocketSession.persist_file
    install_empty_file_upload_compatibility()

    session = object.__new__(WebsocketSession)
    result = await session.persist_file(
        name="script.py",
        mime="text/x-python",
        content=b"",
    )

    assert WebsocketSession.persist_file is installed_method
    assert result == {"id": "empty-file"}
    assert len(received) == 1
    temporary_path, forwarded_content = received[0]
    assert temporary_path is not None
    assert not Path(temporary_path).exists()
    assert forwarded_content is None


@pytest.mark.asyncio
async def test_missing_content_keeps_chainlit_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def chainlit_persist_file(
        self: WebsocketSession,
        name: str,
        mime: str,
        path: str | None = None,
        content: bytes | str | None = None,
    ) -> dict[str, str]:
        del self, name, mime
        if not path and not content:
            raise ValueError("Either path or content must be provided")
        return {"id": "file"}

    monkeypatch.setattr(WebsocketSession, "persist_file", chainlit_persist_file)
    install_empty_file_upload_compatibility()

    session = object.__new__(WebsocketSession)
    with pytest.raises(ValueError, match="Either path or content"):
        await session.persist_file(
            name="script.py",
            mime="text/x-python",
            content=None,
        )


def test_empty_accept_allows_every_spontaneous_upload_without_browser_wildcards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated: list[tuple[object, object | None]] = []

    def chainlit_validate_file_mime_type(file: object, spec: object | None) -> None:
        delegated.append((file, spec))

    monkeypatch.setattr(
        chainlit_server,
        "validate_file_mime_type",
        chainlit_validate_file_mime_type,
    )
    monkeypatch.setattr(config.features.spontaneous_file_upload, "accept", [])
    install_unrestricted_file_upload_compatibility()
    installed_validator = chainlit_server.validate_file_mime_type
    install_unrestricted_file_upload_compatibility()

    upload = object()
    chainlit_server.validate_file_mime_type(upload, None)

    assert chainlit_server.validate_file_mime_type is installed_validator
    assert delegated == []


def test_unrestricted_upload_compatibility_preserves_explicit_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated: list[tuple[object, object | None]] = []

    def chainlit_validate_file_mime_type(file: object, spec: object | None) -> None:
        delegated.append((file, spec))

    monkeypatch.setattr(
        chainlit_server,
        "validate_file_mime_type",
        chainlit_validate_file_mime_type,
    )
    install_unrestricted_file_upload_compatibility()
    upload = object()
    ask_spec = object()

    monkeypatch.setattr(config.features.spontaneous_file_upload, "accept", ["text/*"])
    chainlit_server.validate_file_mime_type(upload, None)
    monkeypatch.setattr(config.features.spontaneous_file_upload, "accept", [])
    chainlit_server.validate_file_mime_type(upload, ask_spec)

    assert delegated == [(upload, None), (upload, ask_spec)]
