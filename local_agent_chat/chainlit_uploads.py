from __future__ import annotations

import inspect
import os
import tempfile
from functools import wraps
from pathlib import Path

import chainlit.server as chainlit_server
from chainlit.config import config
from chainlit.session import WebsocketSession
from chainlit.types import FileReference

_ORIGINAL_METHOD = "_local_agent_chat_original_persist_file"
_EXPECTED_PARAMETERS = ("self", "name", "mime", "path", "content")
_ORIGINAL_MIME_VALIDATOR = "_local_agent_chat_original_mime_validator"
_EXPECTED_MIME_PARAMETERS = ("file", "spec")


def _is_empty_content(content: bytes | str | None) -> bool:
    return (isinstance(content, bytes) and len(content) == 0) or (
        isinstance(content, str) and len(content) == 0
    )


def install_empty_file_upload_compatibility() -> None:
    """Let Chainlit persist valid zero-byte uploads.

    Chainlit 2.11 treats empty bytes as if no content was supplied. Route those
    uploads through its existing path branch so IDs, MIME handling, and session
    metadata remain owned by Chainlit.
    """

    current_method = WebsocketSession.persist_file
    if hasattr(current_method, _ORIGINAL_METHOD):
        return
    if tuple(inspect.signature(current_method).parameters) != _EXPECTED_PARAMETERS:
        raise RuntimeError("Unsupported Chainlit persist_file signature")

    @wraps(current_method)
    async def persist_file(
        self: WebsocketSession,
        name: str,
        mime: str,
        path: str | None = None,
        content: bytes | str | None = None,
    ) -> FileReference:
        if path is not None or not _is_empty_content(content):
            return await current_method(
                self,
                name=name,
                mime=mime,
                path=path,
                content=content,
            )

        descriptor, empty_path = tempfile.mkstemp(
            prefix="local-agent-chat-empty-upload-"
        )
        os.close(descriptor)
        try:
            return await current_method(
                self,
                name=name,
                mime=mime,
                path=empty_path,
                content=None,
            )
        finally:
            Path(empty_path).unlink(missing_ok=True)

    setattr(persist_file, _ORIGINAL_METHOD, current_method)
    WebsocketSession.persist_file = persist_file


def install_unrestricted_file_upload_compatibility() -> None:
    """Treat an empty accept list as unrestricted for spontaneous uploads.

    Chainlit's frontend needs a truthy empty list to avoid substituting its
    invalid ``application/*`` default. Its server currently interprets that
    same list as "reject everything", so align the server with the browser's
    empty accept attribute while preserving explicit and AskFile validation.
    """

    current_validator = chainlit_server.validate_file_mime_type
    if hasattr(current_validator, _ORIGINAL_MIME_VALIDATOR):
        return
    if tuple(inspect.signature(current_validator).parameters) != (
        _EXPECTED_MIME_PARAMETERS
    ):
        raise RuntimeError("Unsupported Chainlit MIME validator signature")

    @wraps(current_validator)
    def validate_file_mime_type(file, spec) -> None:
        upload_feature = config.features.spontaneous_file_upload
        if (
            spec is None
            and upload_feature is not None
            and upload_feature.accept in ([], {})
        ):
            return
        current_validator(file, spec)

    setattr(
        validate_file_mime_type,
        _ORIGINAL_MIME_VALIDATOR,
        current_validator,
    )
    chainlit_server.validate_file_mime_type = validate_file_mime_type
