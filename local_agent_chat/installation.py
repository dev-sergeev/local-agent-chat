"""Installed resources and user-owned directories, independent of the checkout."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"


def config_directory() -> Path:
    return (
        Path(os.environ.get("LOCALCHAT_CONFIG_DIR") or Path.cwd())
        .expanduser()
        .resolve()
    )


def data_directory(directory: Path | None = None) -> Path:
    return ((directory or config_directory()) / ".local-agent-chat").resolve()


def copy_ui(destination: Path) -> None:
    """Copy immutable UI resources where Chainlit may create temporary files."""
    shutil.copytree(ASSETS / "chainlit", destination / ".chainlit")
    shutil.copytree(ASSETS / "public", destination / "public")
    for page in ASSETS.glob("chainlit*.md"):
        shutil.copyfile(page, destination / page.name)


@contextmanager
def runtime_workspace(data_dir: Path) -> Iterator[Path]:
    # Concurrent processes would race the application's cross-database Revision.
    # Keep the lock inode in place even after releasing it.
    import fcntl

    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        data_dir / ".localchat.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                f"Another LocalChat process is using {data_dir}. Stop it first."
            ) from error
        with tempfile.TemporaryDirectory(prefix=".runtime-", dir=data_dir) as root:
            destination = Path(root)
            copy_ui(destination)
            yield destination
