import asyncio
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from local_agent_chat.sandbox_files import SandboxFiles


class SandboxFilesTest(unittest.IsolatedAsyncioTestCase):
    async def test_upload_keeps_files_with_the_same_basename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_source = root / "first" / "report.txt"
            second_source = root / "second" / "report.txt"
            first_source.parent.mkdir()
            second_source.parent.mkdir()
            first_source.write_text("first report", encoding="utf-8")
            second_source.write_text("second report", encoding="utf-8")
            files = SandboxFiles(
                root / "sandboxes", max_file_bytes=100, max_chat_bytes=200
            )

            first = await files.upload("chat-1", first_source, first_source.name)
            second = await files.upload("chat-1", second_source, second_source.name)

            self.assertEqual(first.name, "report.txt")
            self.assertEqual(second.name, "report (2).txt")
            self.assertEqual(first.read_text(encoding="utf-8"), "first report")
            self.assertEqual(second.read_text(encoding="utf-8"), "second report")
            self.assertEqual(
                set(files.manifest("chat-1")), {"report.txt", "report (2).txt"}
            )

    async def test_concurrent_uploads_serialize_names_and_chat_quota(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for index in range(6):
                source = root / f"source-{index}" / "report.txt"
                source.parent.mkdir()
                source.write_text(str(index), encoding="utf-8")
                sources.append(source)
            files = SandboxFiles(
                root / "sandboxes", max_file_bytes=10, max_chat_bytes=10
            )

            stored = await asyncio.gather(
                *(files.upload("chat-1", source, source.name) for source in sources)
            )

            self.assertEqual(len({path.name for path in stored}), len(sources))
            self.assertEqual(
                {path.read_text(encoding="utf-8") for path in stored},
                {str(index) for index in range(6)},
            )

            extra_a = root / "extra-a"
            extra_b = root / "extra-b"
            extra_a.write_bytes(b"123456")
            extra_b.write_bytes(b"789012")
            outcomes = await asyncio.gather(
                files.upload("chat-2", extra_a, "extra-a"),
                files.upload("chat-2", extra_b, "extra-b"),
                return_exceptions=True,
            )
            self.assertEqual(sum(isinstance(item, Path) for item in outcomes), 1)
            self.assertEqual(sum(isinstance(item, ValueError) for item in outcomes), 1)
            self.assertEqual(len(files.manifest("chat-2")), 1)

    async def test_failed_replacement_never_publishes_a_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.txt"
            replacement = root / "replacement.txt"
            original.write_text("original", encoding="utf-8")
            replacement.write_text("replacement", encoding="utf-8")
            files = SandboxFiles(
                root / "sandboxes", max_file_bytes=100, max_chat_bytes=200
            )
            stored = await files.upload("chat-1", original, "report.txt")

            def fail_copy(_source: Path, destination: Path) -> None:
                destination.write_text("partial", encoding="utf-8")
                raise OSError("disk full")

            with (
                patch(
                    "local_agent_chat.sandbox_files.shutil.copy2",
                    side_effect=fail_copy,
                ),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                await files.upload("chat-1", replacement, "report.txt", replace=True)

            self.assertEqual(stored.read_text(encoding="utf-8"), "original")
            staging = root / "sandboxes" / "chat-1" / "staging"
            self.assertEqual(list(staging.iterdir()), [])

    async def test_uploaded_file_is_restored_from_pre_turn_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "upload.txt"
            source.write_text("original", encoding="utf-8")
            files = SandboxFiles(
                root / "sandboxes", max_file_bytes=100, max_chat_bytes=200
            )

            stored = await files.upload("chat-1", source, "../report.txt")
            artifact = files.artifacts_dir("chat-1") / "conversation_history" / "1"
            artifact.parent.mkdir()
            artifact.write_text("original context", encoding="utf-8")
            snapshot = await files.snapshot("chat-1")
            stored.write_text("changed after snapshot", encoding="utf-8")
            artifact.write_text("changed context", encoding="utf-8")
            await files.restore("chat-1", snapshot)

            self.assertEqual(stored.read_text(encoding="utf-8"), "original")
            self.assertEqual(stored.parent, root / "sandboxes" / "chat-1" / "files")
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

    async def test_failed_restore_keeps_both_current_trees(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = SandboxFiles(
                root / "sandboxes", max_file_bytes=100, max_chat_bytes=200
            )
            user_file = files.files_dir("chat-1") / "report.txt"
            artifact = files.artifacts_dir("chat-1") / "context.txt"
            user_file.write_text("snapshot user", encoding="utf-8")
            artifact.write_text("snapshot artifact", encoding="utf-8")
            snapshot = await files.snapshot("chat-1")
            user_file.write_text("current user", encoding="utf-8")
            artifact.write_text("current artifact", encoding="utf-8")
            copytree = shutil.copytree
            calls = 0

            def fail_second_copy(source: Path, destination: Path, **kwargs) -> Path:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("disk full")
                return copytree(source, destination, **kwargs)

            with (
                patch(
                    "local_agent_chat.sandbox_files.shutil.copytree",
                    side_effect=fail_second_copy,
                ),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                await files.restore("chat-1", snapshot)

            self.assertEqual(user_file.read_text(encoding="utf-8"), "current user")
            self.assertEqual(artifact.read_text(encoding="utf-8"), "current artifact")

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

    async def test_replacing_file_counts_only_the_replacement_toward_chat_limit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.txt"
            replacement = root / "replacement.txt"
            original.write_bytes(b"1234")
            replacement.write_bytes(b"12345")
            files = SandboxFiles(root / "sandboxes", max_file_bytes=5, max_chat_bytes=5)
            await files.upload("chat-1", original, "report.txt")

            stored = await files.upload(
                "chat-1", replacement, "report.txt", replace=True
            )

            self.assertEqual(stored.name, "report.txt")
            self.assertEqual(stored.read_bytes(), b"12345")
            self.assertEqual(set(files.manifest("chat-1")), {"report.txt"})

    async def test_deleting_chat_removes_files_snapshots_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = SandboxFiles(
                root / "sandboxes", max_file_bytes=100, max_chat_bytes=200
            )
            files.files_dir("chat-1").joinpath("file.txt").write_text("content")
            files.artifacts_dir("chat-1").joinpath("marker").write_text("context")
            await files.snapshot("chat-1")
            await files.delete_chat("chat-1")
            self.assertFalse((root / "sandboxes" / "chat-1").exists())


if __name__ == "__main__":
    unittest.main()
