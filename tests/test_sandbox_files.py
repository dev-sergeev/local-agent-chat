import tempfile
import unittest
import uuid
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
            environment = files.environment_dir("chat-1")
            environment.joinpath("installed-package").write_text("keep")
            artifact = files.artifacts_dir("chat-1") / "conversation_history" / "1"
            artifact.parent.mkdir()
            artifact.write_text("original context", encoding="utf-8")
            snapshot = await files.snapshot("chat-1")
            stored.write_text("changed by agent", encoding="utf-8")
            artifact.write_text("changed context", encoding="utf-8")
            await files.restore("chat-1", snapshot)

            self.assertEqual(stored.read_text(encoding="utf-8"), "original")
            self.assertEqual(stored.parent, root / "sandboxes" / "chat-1" / "files")
            self.assertEqual(
                environment.joinpath("installed-package").read_text(), "keep"
            )
            self.assertEqual(artifact.read_text(), "original context")
            self.assertEqual(set(files.manifest("chat-1")), {"report.txt"})

    async def test_legacy_file_only_snapshot_restores_and_clears_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = SandboxFiles(
                root / "sandboxes", max_file_bytes=100, max_chat_bytes=200
            )
            files.files_dir("chat-1").joinpath("current.txt").write_text("current")
            files.artifacts_dir("chat-1").joinpath("stale.txt").write_text("stale")
            snapshot = uuid.uuid4().hex
            legacy = root / "sandboxes" / "chat-1" / "snapshots" / snapshot
            legacy.mkdir(parents=True)
            legacy.joinpath("legacy.txt").write_text("restored")

            await files.restore("chat-1", snapshot)

            self.assertEqual(
                files.files_dir("chat-1").joinpath("legacy.txt").read_text(),
                "restored",
            )
            self.assertFalse(files.files_dir("chat-1").joinpath("current.txt").exists())
            self.assertEqual(list(files.artifacts_dir("chat-1").iterdir()), [])

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

    async def test_deleting_chat_removes_files_snapshots_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = SandboxFiles(
                root / "sandboxes", max_file_bytes=100, max_chat_bytes=200
            )
            files.files_dir("chat-1").joinpath("file.txt").write_text("content")
            files.environment_dir("chat-1").joinpath("marker").write_text("runtime")
            await files.snapshot("chat-1")
            await files.delete_chat("chat-1")
            self.assertFalse((root / "sandboxes" / "chat-1").exists())


if __name__ == "__main__":
    unittest.main()
