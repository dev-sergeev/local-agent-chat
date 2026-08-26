import asyncio
import tempfile
import unittest
from pathlib import Path

from local_agent_chat.agent_events import EventSink, TextDelta
from local_agent_chat.runtime import ChatRuntime, Turn
from local_agent_chat.sqlite_history import SQLiteHistory


class RecordingAgent:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def checkpoint(self, chat_id: str) -> str:
        return "memory-before-first"

    async def restore(self, chat_id: str, checkpoint: str) -> None:
        self.events.append(("restore-memory", checkpoint))

    async def run(self, chat_id: str, text: str, emit: EventSink | None = None) -> str:
        self.events.append(("run", text))
        if emit is not None:
            await emit(TextDelta(f"answer:{text}"))
        return f"answer:{text}"


class RecordingSandbox:
    def __init__(self, events: list[tuple[str, str]]) -> None:
        self.events = events

    async def snapshot(self, chat_id: str) -> str:
        return "files-before-first"

    async def restore(self, chat_id: str, snapshot: str) -> None:
        self.events.append(("restore-sandbox", snapshot))


class InMemoryHistory:
    def __init__(self, events: list[tuple[str, str]]) -> None:
        self.events = events
        self.turns: list[Turn] = []

    async def append(self, turn: Turn) -> None:
        self.turns.append(turn)

    async def replace_from(self, turn_id: str, turn: Turn) -> None:
        self.events.append(("replace-history", turn.text))
        index = next(i for i, item in enumerate(self.turns) if item.id == turn_id)
        self.turns[index:] = [turn]

    async def get(self, turn_id: str) -> Turn:
        return next(item for item in self.turns if item.id == turn_id)


class ChatRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_turn_for_same_chat_is_rejected(self) -> None:
        agent = RecordingAgent()
        history = InMemoryHistory(agent.events)
        runtime = ChatRuntime(
            agent=agent, sandbox=RecordingSandbox(agent.events), history=history
        )
        started = asyncio.Event()
        release = asyncio.Event()

        async def wait(chat_id: str, text: str, emit: EventSink | None = None) -> str:
            started.set()
            await release.wait()
            return "done"

        agent.run = wait  # type: ignore[method-assign]
        first = asyncio.create_task(runtime.submit("chat-1", "turn-1", "first"))
        await started.wait()
        with self.assertRaisesRegex(RuntimeError, "already running"):
            await runtime.submit("chat-1", "turn-2", "second")
        release.set()
        await first

    async def test_revision_restores_both_states_then_reruns_agent(self) -> None:
        agent = RecordingAgent()
        history = InMemoryHistory(agent.events)
        sandbox = RecordingSandbox(agent.events)
        runtime = ChatRuntime(agent=agent, sandbox=sandbox, history=history)

        await runtime.submit("chat-1", "turn-1", "original")
        await runtime.submit("chat-1", "turn-2", "later")
        agent.events.clear()

        result = await runtime.revise("chat-1", "turn-1", "revised")

        self.assertEqual(result, "answer:revised")
        self.assertEqual([turn.text for turn in history.turns], ["revised"])
        self.assertEqual(
            agent.events[:4],
            [
                ("restore-memory", "memory-before-first"),
                ("restore-sandbox", "files-before-first"),
                ("run", "revised"),
                ("replace-history", "revised"),
            ],
        )

    async def test_failed_revision_keeps_active_history_unchanged(self) -> None:
        agent = RecordingAgent()
        history = InMemoryHistory(agent.events)
        runtime = ChatRuntime(
            agent=agent, sandbox=RecordingSandbox(agent.events), history=history
        )
        await runtime.submit("chat-1", "turn-1", "original")

        async def fail(chat_id: str, text: str, emit: EventSink | None = None) -> str:
            raise RuntimeError("model failed")

        agent.run = fail  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            await runtime.revise("chat-1", "turn-1", "revised")

        self.assertEqual(
            (history.turns[0].text, history.turns[0].answer),
            ("original", "answer:original"),
        )

    async def test_submit_rolls_back_when_history_persistence_fails(self) -> None:
        agent = RecordingAgent()

        class FailingHistory(InMemoryHistory):
            async def append(self, turn: Turn) -> None:
                del turn
                raise RuntimeError("history unavailable")

        history = FailingHistory(agent.events)
        runtime = ChatRuntime(
            agent=agent, sandbox=RecordingSandbox(agent.events), history=history
        )

        with self.assertRaisesRegex(RuntimeError, "history unavailable"):
            await runtime.submit("chat-1", "turn-1", "request")

        self.assertEqual(history.turns, [])
        self.assertEqual(
            agent.events[-2:],
            [
                ("restore-memory", "memory-before-first"),
                ("restore-sandbox", "files-before-first"),
            ],
        )

    async def test_revision_rolls_back_when_history_replacement_fails(self) -> None:
        agent = RecordingAgent()

        class FailingHistory(InMemoryHistory):
            async def replace_from(self, turn_id: str, turn: Turn) -> None:
                del turn_id, turn
                raise RuntimeError("history unavailable")

        history = FailingHistory(agent.events)
        runtime = ChatRuntime(
            agent=agent, sandbox=RecordingSandbox(agent.events), history=history
        )
        await runtime.submit("chat-1", "turn-1", "original")
        agent.events.clear()

        with self.assertRaisesRegex(RuntimeError, "history unavailable"):
            await runtime.revise("chat-1", "turn-1", "revised")

        self.assertEqual(history.turns[0].text, "original")
        self.assertEqual(
            agent.events[-2:],
            [
                ("restore-memory", "memory-before-first"),
                ("restore-sandbox", "files-before-first"),
            ],
        )

    async def test_submit_forwards_events_and_rolls_back_cancelled_turn(self) -> None:
        agent = RecordingAgent()
        history = InMemoryHistory(agent.events)
        sandbox = RecordingSandbox(agent.events)
        runtime = ChatRuntime(agent=agent, sandbox=sandbox, history=history)
        seen: list[TextDelta] = []

        async def cancel(chat_id: str, text: str, emit: EventSink | None = None) -> str:
            if emit is not None:
                await emit(TextDelta("partial"))
            raise asyncio.CancelledError

        agent.run = cancel  # type: ignore[method-assign]

        async def record(event) -> None:
            seen.append(event)

        with self.assertRaises(asyncio.CancelledError):
            await runtime.submit("chat-1", "turn-1", "cancel", record)

        self.assertEqual(seen, [TextDelta("partial")])
        self.assertEqual(history.turns, [])
        self.assertEqual(
            agent.events,
            [
                ("restore-memory", "memory-before-first"),
                ("restore-sandbox", "files-before-first"),
            ],
        )

    async def test_rollback_attempts_both_restores_when_one_fails(self) -> None:
        agent = RecordingAgent()
        history = InMemoryHistory(agent.events)
        sandbox = RecordingSandbox(agent.events)
        runtime = ChatRuntime(agent=agent, sandbox=sandbox, history=history)

        async def fail_run(
            chat_id: str, text: str, emit: EventSink | None = None
        ) -> str:
            raise RuntimeError("model failed")

        async def fail_memory_restore(chat_id: str, checkpoint: str) -> None:
            agent.events.append(("restore-memory", checkpoint))
            raise RuntimeError("memory restore failed")

        agent.run = fail_run  # type: ignore[method-assign]
        agent.restore = fail_memory_restore  # type: ignore[method-assign]

        with self.assertRaisesRegex(RuntimeError, "memory restore failed"):
            await runtime.submit("chat-1", "turn-1", "request")

        self.assertEqual(
            agent.events,
            [
                ("restore-memory", "memory-before-first"),
                ("restore-sandbox", "files-before-first"),
            ],
        )

    async def test_revised_history_survives_repository_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "history.sqlite3"
            agent = RecordingAgent()
            history = SQLiteHistory(database)
            runtime = ChatRuntime(
                agent=agent,
                sandbox=RecordingSandbox(agent.events),
                history=history,
            )
            await runtime.submit("chat-1", "turn-1", "original")
            await runtime.submit("chat-1", "turn-2", "later")
            await runtime.revise("chat-1", "turn-1", "revised")

            reopened = SQLiteHistory(database)
            revised = await reopened.get("turn-1")

            self.assertEqual(
                (revised.text, revised.answer), ("revised", "answer:revised")
            )
            with self.assertRaises(KeyError):
                await reopened.get("turn-2")

    async def test_sqlite_history_reports_whether_chat_has_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history = SQLiteHistory(Path(directory) / "history.sqlite3")

            self.assertFalse(await history.has_chat("chat-1"))
            await history.append(
                Turn("turn-1", "chat-1", "request", "answer", "memory", "files")
            )
            self.assertTrue(await history.has_chat("chat-1"))

    async def test_delete_waits_for_pre_run_and_post_run_transaction_stages(
        self,
    ) -> None:
        agent = RecordingAgent()
        snapshot_started = asyncio.Event()
        release_snapshot = asyncio.Event()
        append_started = asyncio.Event()
        release_append = asyncio.Event()
        cleanup_called = asyncio.Event()

        class BlockingSandbox(RecordingSandbox):
            async def snapshot(self, chat_id: str) -> str:
                snapshot_started.set()
                await release_snapshot.wait()
                return await super().snapshot(chat_id)

        class BlockingHistory(InMemoryHistory):
            async def append(self, turn: Turn) -> None:
                append_started.set()
                await release_append.wait()
                await super().append(turn)

        history = BlockingHistory(agent.events)
        runtime = ChatRuntime(
            agent=agent,
            sandbox=BlockingSandbox(agent.events),
            history=history,
        )
        running = asyncio.create_task(runtime.submit("chat-1", "turn-1", "request"))
        await snapshot_started.wait()

        async def cleanup() -> None:
            cleanup_called.set()

        deleting = asyncio.create_task(runtime.delete_chat("chat-1", cleanup))
        await asyncio.sleep(0)
        self.assertFalse(deleting.done())
        release_snapshot.set()
        await append_started.wait()
        self.assertFalse(deleting.done())
        self.assertFalse(cleanup_called.is_set())
        release_append.set()

        self.assertEqual(await running, "answer:request")
        await deleting
        self.assertTrue(cleanup_called.is_set())
        with self.assertRaisesRegex(RuntimeError, "being deleted"):
            await runtime.submit("chat-1", "turn-2", "later")


if __name__ == "__main__":
    unittest.main()
