from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from .agent_events import EventSink


@dataclass(frozen=True)
class Turn:
    id: str
    chat_id: str
    text: str
    answer: str
    memory_checkpoint: str
    sandbox_snapshot: str


class Agent(Protocol):
    async def checkpoint(self, chat_id: str) -> str: ...
    async def restore(self, chat_id: str, checkpoint: str) -> None: ...
    async def run(
        self, chat_id: str, text: str, emit: EventSink | None = None
    ) -> str: ...


class Sandbox(Protocol):
    async def snapshot(self, chat_id: str) -> str: ...
    async def restore(self, chat_id: str, snapshot: str) -> None: ...


class History(Protocol):
    async def append(self, turn: Turn) -> None: ...
    async def replace_from(self, turn_id: str, turn: Turn) -> None: ...
    async def get(self, turn_id: str) -> Turn: ...
    async def set_answer(self, turn_id: str, answer: str) -> None: ...


class ChatRuntime:
    def __init__(self, *, agent: Agent, sandbox: Sandbox, history: History) -> None:
        self._agent = agent
        self._sandbox = sandbox
        self._history = history
        self._locks: dict[str, asyncio.Lock] = {}

    def _enter_chat(self, chat_id: str) -> asyncio.Lock:
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        if lock.locked():
            raise RuntimeError("Another Turn is already running for this Chat")
        return lock

    async def _restore_state(self, chat_id: str, memory: str, files: str) -> None:
        async def restore() -> None:
            await self._agent.restore(chat_id, memory)
            await self._sandbox.restore(chat_id, files)

        await asyncio.shield(restore())

    async def submit(
        self,
        chat_id: str,
        turn_id: str,
        text: str,
        emit: EventSink | None = None,
    ) -> str:
        async with self._enter_chat(chat_id):
            memory = await self._agent.checkpoint(chat_id)
            files = await self._sandbox.snapshot(chat_id)
            try:
                answer = await self._agent.run(chat_id, text, emit)
            except (Exception, asyncio.CancelledError):
                await self._restore_state(chat_id, memory, files)
                raise
            await self._history.append(
                Turn(turn_id, chat_id, text, answer, memory, files)
            )
            return answer

    async def has_turn(self, turn_id: str) -> bool:
        try:
            await self._history.get(turn_id)
        except KeyError:
            return False
        return True

    async def revise(
        self,
        chat_id: str,
        turn_id: str,
        text: str,
        emit: EventSink | None = None,
    ) -> str:
        async with self._enter_chat(chat_id):
            original = await self._history.get(turn_id)
            if original.chat_id != chat_id:
                raise ValueError("Turn does not belong to Chat")

            rollback_memory = await self._agent.checkpoint(chat_id)
            rollback_files = await self._sandbox.snapshot(chat_id)
            try:
                await self._agent.restore(chat_id, original.memory_checkpoint)
                await self._sandbox.restore(chat_id, original.sandbox_snapshot)
                answer = await self._agent.run(chat_id, text, emit)
            except (Exception, asyncio.CancelledError):
                await self._restore_state(chat_id, rollback_memory, rollback_files)
                raise
            replacement = Turn(
                turn_id,
                chat_id,
                text,
                answer,
                original.memory_checkpoint,
                original.sandbox_snapshot,
            )
            await self._history.replace_from(turn_id, replacement)
            return answer
