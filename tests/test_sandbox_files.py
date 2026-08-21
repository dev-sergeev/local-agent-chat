import tempfile
import unittest
from pathlib import Path

from local_agent_chat.sandbox_files import SandboxFiles


class SandboxFilesTest(unittest.IsolatedAsyncioTestCase):
    async def test_uploaded_file_is_restored_from_pre_turn_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "upload.txt"
            source.write_text("original", encoding="utf-8")
            files = SandboxFiles(
                root / "sandboxes", max_file_bytes=100, max_chat_bytes=200
            )

            stored = await files.upload("chat-1", source, "../report.txt")
            snapshot = await files.snapshot("chat-1")
            stored.write_text("changed by agent", encoding="utf-8")
            await files.restore("chat-1", snapshot)

            self.assertEqual(stored.read_text(encoding="utf-8"), "original")
            self.assertEqual(stored.parent, root / "sandboxes" / "chat-1" / "files")

    async def test_rejects_file_before_copy_when_limit_is_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "large.bin"
            source.write_bytes(b"12345")
            files = SandboxFiles(
                root / "sandboxes", max_file_bytes=4, max_chat_bytes=10
            )

            with self.assertRaisesRegex(ValueError, "file limit"):
                await files.upload("chat-1", source, "large.bin")

            self.assertFalse(
                (root / "sandboxes" / "chat-1" / "files" / "large.bin").exists()
            )

    async def test_deleting_chat_removes_files_and_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = SandboxFiles(
                root / "sandboxes", max_file_bytes=100, max_chat_bytes=200
            )
            files.files_dir("chat-1").joinpath("file.txt").write_text("content")
            await files.snapshot("chat-1")
            await files.delete_chat("chat-1")
            self.assertFalse((root / "sandboxes" / "chat-1").exists())


if __name__ == "__main__":
    unittest.main()
