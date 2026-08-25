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

    async def set_answer(self, turn_id: str, answer: str) -> None:
        index = next(i for i, item in enumerate(self.turns) if item.id == turn_id)
        current = self.turns[index]
        self.turns[index] = Turn(
            current.id,
            current.chat_id,
            current.text,
            answer,
            current.memory_checkpoint,
            current.sandbox_snapshot,
        )


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


if __name__ == "__main__":
    unittest.main()
