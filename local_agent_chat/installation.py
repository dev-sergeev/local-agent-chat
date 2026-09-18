"""Installed resources and user-owned directories, independent of the checkout."""

from __future__ import annotations

import errno
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


def data_directory(
    directory: Path | None = None, *, value: str | Path | None = None
) -> Path:
    """Resolve data beside .env (the launch directory by default), before use."""
    path = Path(value or ".local-agent-chat").expanduser()
    if not path.is_absolute():
        path = (directory or config_directory()) / path
    return path.resolve()


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
        except OSError as error:
            if error.errno not in {errno.ENOLCK, errno.EOPNOTSUPP, errno.ENOSYS}:
                raise
        # Atomic mkdir also works when the filesystem has no flock service.
        # All instances take this guard, even when flock succeeds, so clients
        # with different locking support cannot open the same data concurrently.
        guard = data_dir / ".localchat.lock.d"
        try:
            guard.mkdir(mode=0o700)
        except FileExistsError as error:
            raise ValueError(
                f"Another LocalChat process may be using {data_dir}. Stop it first. "
                f"If all instances are stopped, remove the stale lock directory {guard}."
            ) from error
        try:
            with tempfile.TemporaryDirectory(prefix=".runtime-", dir=data_dir) as root:
                destination = Path(root)
                copy_ui(destination)
                yield destination
        finally:
            guard.rmdir()
