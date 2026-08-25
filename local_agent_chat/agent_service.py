from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import aiosqlite
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .agent_events import (
    EventSink,
    TextDelta,
    ToolFailed,
    ToolFinished,
    ToolStarted,
    message_text,
    public_text,
    safe_text,
)
from .prompts import AGENT_SYSTEM_PROMPT, TOOL_TITLE_SYSTEM_PROMPT
from .sandbox_provider import LocalSandboxManager
from .settings import ModelProfile
from .tool_titles import normalize_tool_title


class AgentService:
    def __init__(
        self,
        database: Path,
        models: tuple[ModelProfile, ...],
        sandboxes: LocalSandboxManager,
    ) -> None:
        self._database = database
        self._models = {model.id: model for model in models}
        self._sandboxes = sandboxes
        self._connection: aiosqlite.Connection | None = None
        self._saver: AsyncSqliteSaver | None = None
        self._graphs: dict[str, Any] = {}
        self._tool_title_models: dict[str, Any] = {}
        self._profiles: dict[str, str] = {}
        self._namespaces: dict[str, str] = {}
        self._pending_restore: dict[str, dict[str, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._init_registry()

    def _init_registry(self) -> None:
        self._database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._database) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS active_branches (chat_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL)"
            )
            for chat_id, profile_id, namespace in connection.execute(
                "SELECT chat_id, profile_id, checkpoint_ns FROM active_branches"
            ):
                self._profiles[chat_id] = profile_id
                self._namespaces[chat_id] = namespace

    async def _checkpointer(self) -> AsyncSqliteSaver:
        if self._saver is None:
            self._connection = await aiosqlite.connect(self._database)
            self._saver = AsyncSqliteSaver(self._connection)
            await self._saver.setup()
        return self._saver

    def set_profile(self, chat_id: str, profile_id: str) -> None:
        if profile_id not in self._models:
            raise ValueError(f"Unknown Model Profile: {profile_id}")
        existing = self._profiles.get(chat_id)
        if existing and existing != profile_id:
            raise ValueError("Model Profile cannot change inside an existing Chat")
        namespace = self._namespaces.setdefault(chat_id, "")
        self._profiles[chat_id] = profile_id
        with sqlite3.connect(self._database) as connection:
            connection.execute(
                "INSERT INTO active_branches(chat_id, profile_id, checkpoint_ns) VALUES (?, ?, ?) ON CONFLICT(chat_id) DO UPDATE SET profile_id=excluded.profile_id, checkpoint_ns=excluded.checkpoint_ns",
                (chat_id, profile_id, namespace),
            )

    def profile_for(self, chat_id: str) -> str | None:
        return self._profiles.get(chat_id)

    async def describe_tool(
        self, chat_id: str, name: str, input_text: str
    ) -> str | None:
        """Generate a compact label without affecting the agent's main run."""

        profile_id = self._profiles.get(chat_id)
        if profile_id is None:
            return None
        profile = self._models[profile_id]
        model = self._tool_title_models.get(profile_id)
        if model is None:
            kwargs: dict[str, Any] = {
                "max_tokens": 32,
                "reasoning_effort": "none",
            }
            if profile.api_key:
                kwargs["api_key"] = profile.api_key
            if profile.base_url:
                kwargs["base_url"] = profile.base_url
            model = init_chat_model(profile.model, **kwargs)
            self._tool_title_models[profile_id] = model
        payload = safe_text(input_text, max_chars=2000)
        try:
            response = await asyncio.wait_for(
                model.ainvoke(
                    [
                        ("system", TOOL_TITLE_SYSTEM_PROMPT),
                        (
                            "user",
                            f"Инструмент: {name}\n<tool-input>\n{payload}\n</tool-input>",
                        ),
                    ]
                ),
                timeout=10,
            )
        except Exception:  # noqa: BLE001 - a cosmetic label must never fail a Turn
            return None
        return normalize_tool_title(message_text(response))

    async def _graph(self, chat_id: str):
        if chat_id in self._graphs:
            return self._graphs[chat_id]
        profile = self._models[self._profiles[chat_id]]
        kwargs: dict[str, Any] = {}
        if profile.api_key:
            kwargs["api_key"] = profile.api_key
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        model = init_chat_model(profile.model, **kwargs)
        backend = await self._sandboxes.backend(chat_id)
        graph = create_deep_agent(
            model=model,
            backend=backend,
            checkpointer=await self._checkpointer(),
            system_prompt=AGENT_SYSTEM_PROMPT,
        )
        self._graphs[chat_id] = graph
        return graph

    def _base_config(self, chat_id: str) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": chat_id,
                "checkpoint_ns": self._namespaces.get(chat_id, ""),
            }
        }

    async def checkpoint(self, chat_id: str) -> str:
        saver = await self._checkpointer()
        config = self._base_config(chat_id)
        item = await saver.aget_tuple(config)
        checkpoint_id = (
            item.config["configurable"].get("checkpoint_id") if item else None
        )
        return json.dumps(
            {
                "checkpoint_ns": config["configurable"]["checkpoint_ns"],
                "checkpoint_id": checkpoint_id,
            }
        )

    async def restore(self, chat_id: str, checkpoint: str) -> None:
        token = json.loads(checkpoint)
        if token.get("checkpoint_id"):
            self._pending_restore[chat_id] = {
                "configurable": {"thread_id": chat_id, **token}
            }
        else:
            namespace = uuid.uuid4().hex
            self._namespaces[chat_id] = namespace
            self._pending_restore[chat_id] = {
                "configurable": {"thread_id": chat_id, "checkpoint_ns": namespace}
            }
            self.set_profile(chat_id, self._profiles[chat_id])

    async def run(self, chat_id: str, text: str, emit: EventSink | None = None) -> str:
        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            graph = await self._graph(chat_id)
            backend = await self._sandboxes.backend(chat_id)
            await self._sandboxes.push(chat_id, backend)
            config = self._pending_restore.pop(chat_id, self._base_config(chat_id))
            result: dict[str, Any] | None = None
            try:
                async for event in graph.astream_events(
                    {"messages": [{"role": "user", "content": text}]},
                    config=config,
                    version="v2",
                ):
                    event_type = event.get("event")
                    data = event.get("data", {})
                    run_id = str(event.get("run_id", ""))
                    if event_type == "on_tool_start" and emit is not None:
                        await emit(
                            ToolStarted(
                                id=run_id,
                                name=str(event.get("name") or "tool"),
                                input=safe_text(data.get("input"), max_chars=4000),
                            )
                        )
                    elif event_type == "on_tool_end" and emit is not None:
                        output = data.get("output")
                        await emit(
                            ToolFinished(
                                id=run_id,
                                output=safe_text(
                                    getattr(output, "content", output), max_chars=6000
                                ),
                            )
                        )
                    elif event_type == "on_tool_error" and emit is not None:
                        await emit(
                            ToolFailed(
                                id=run_id,
                                error=safe_text(data.get("error"), max_chars=4000),
                            )
                        )
                    elif event_type == "on_chat_model_stream" and emit is not None:
                        chunk = data.get("chunk")
                        if getattr(chunk, "tool_call_chunks", None):
                            continue
                        delta = public_text(getattr(chunk, "content", None))
                        if delta:
                            await emit(TextDelta(delta))
                    elif (
                        event_type == "on_chain_end"
                        and not event.get("parent_ids")
                        and isinstance(data.get("output"), dict)
                        and "messages" in data["output"]
                    ):
                        result = data["output"]
            finally:
                await self._sandboxes.pull(chat_id, backend)
            if result is None:
                raise RuntimeError("Agent finished without a final state")
            message = result["messages"][-1]
            return message_text(message)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()

    async def delete_chat(self, chat_id: str) -> None:
        saver = await self._checkpointer()
        await saver.adelete_thread(chat_id)
        self._profiles.pop(chat_id, None)
        self._namespaces.pop(chat_id, None)
        self._pending_restore.pop(chat_id, None)
        self._graphs.pop(chat_id, None)
        with sqlite3.connect(self._database) as connection:
            connection.execute(
                "DELETE FROM active_branches WHERE chat_id = ?", (chat_id,)
            )
