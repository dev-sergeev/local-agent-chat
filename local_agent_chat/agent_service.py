from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import aiosqlite
from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
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
from .agent_modes import (
    EXTENDED_FILESYSTEM_TOOLS,
    READ_ONLY_FILESYSTEM_TOOLS,
    AgentMode,
)
from .chat_titles import normalize_chat_title
from .memory_tools import build_global_memory_tools
from .prompts import (
    CHAT_TITLE_SYSTEM_PROMPT,
    TOOL_TITLE_RETRY_PROMPT,
    TOOL_TITLE_SYSTEM_PROMPT,
    agent_system_prompt,
)
from .sandbox_provider import LocalSandboxManager
from .settings import ModelProfile
from .sqlite_history import SQLiteHistory
from .tool_titles import normalize_tool_title


class AgentService:
    def __init__(
        self,
        database: Path,
        models: tuple[ModelProfile, ...],
        sandboxes: LocalSandboxManager,
        global_memory: SQLiteHistory | None = None,
    ) -> None:
        self._database = database
        self._models = {model.id: model for model in models}
        self._sandboxes = sandboxes
        self._global_memory = global_memory
        self._connection: aiosqlite.Connection | None = None
        self._saver: AsyncSqliteSaver | None = None
        self._graphs: dict[str, Any] = {}
        self._title_models: dict[str, Any] = {}
        self._profiles: dict[str, str] = {}
        self._modes: dict[str, AgentMode] = {}
        self._mode_locks: dict[str, bool] = {}
        self._namespaces: dict[str, str] = {}
        self._pending_restore: dict[str, dict[str, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._deleting: set[str] = set()
        self._init_registry()

    def _init_registry(self) -> None:
        self._database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._database) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS active_branches (
                       chat_id TEXT PRIMARY KEY,
                       profile_id TEXT NOT NULL,
                       checkpoint_ns TEXT NOT NULL,
                       agent_mode TEXT NOT NULL DEFAULT 'read_only'
                           CHECK(agent_mode IN ('read_only', 'extended')),
                       mode_locked INTEGER NOT NULL DEFAULT 0
                           CHECK(mode_locked IN (0, 1))
                   )"""
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(active_branches)")
            }
            legacy_registry = "agent_mode" not in columns
            if legacy_registry:
                connection.execute(
                    "ALTER TABLE active_branches ADD COLUMN agent_mode TEXT NOT NULL DEFAULT 'read_only'"
                )
            if "mode_locked" not in columns:
                connection.execute(
                    "ALTER TABLE active_branches ADD COLUMN mode_locked INTEGER NOT NULL DEFAULT 0"
                )
            if legacy_registry:
                connection.execute(
                    "UPDATE active_branches SET agent_mode = 'extended', mode_locked = 1"
                )
            invalid_rows: list[str] = []
            for (
                chat_id,
                profile_id,
                namespace,
                raw_mode,
                mode_locked,
            ) in connection.execute(
                "SELECT chat_id, profile_id, checkpoint_ns, agent_mode, mode_locked FROM active_branches"
            ):
                self._profiles[chat_id] = profile_id
                self._namespaces[chat_id] = namespace
                try:
                    mode = AgentMode(raw_mode)
                except ValueError:
                    mode = AgentMode.READ_ONLY
                    invalid_rows.append(chat_id)
                self._modes[chat_id] = mode
                self._mode_locks[chat_id] = bool(mode_locked)
            connection.executemany(
                "UPDATE active_branches SET agent_mode = 'read_only' WHERE chat_id = ?",
                ((chat_id,) for chat_id in invalid_rows),
            )

    async def _checkpointer(self) -> AsyncSqliteSaver:
        if self._saver is None:
            self._connection = await aiosqlite.connect(self._database)
            self._saver = AsyncSqliteSaver(self._connection)
            await self._saver.setup()
        return self._saver

    def set_profile(self, chat_id: str, profile_id: str) -> None:
        if chat_id in self._deleting:
            raise RuntimeError("Chat is being deleted")
        if profile_id not in self._models:
            raise ValueError(f"Unknown Model Profile: {profile_id}")
        existing = self._profiles.get(chat_id)
        if existing and existing != profile_id:
            raise ValueError("Model Profile cannot change inside an existing Chat")
        namespace = self._namespaces.setdefault(chat_id, "")
        self._profiles[chat_id] = profile_id
        self._modes.setdefault(chat_id, AgentMode.READ_ONLY)
        self._mode_locks.setdefault(chat_id, False)
        with sqlite3.connect(self._database) as connection:
            connection.execute(
                "INSERT INTO active_branches(chat_id, profile_id, checkpoint_ns) VALUES (?, ?, ?) ON CONFLICT(chat_id) DO UPDATE SET profile_id=excluded.profile_id, checkpoint_ns=excluded.checkpoint_ns",
                (chat_id, profile_id, namespace),
            )

    def profile_for(self, chat_id: str) -> str | None:
        return self._profiles.get(chat_id)

    def mode_for(self, chat_id: str) -> AgentMode | None:
        return self._modes.get(chat_id)

    def mode_is_locked(self, chat_id: str) -> bool:
        return self._mode_locks.get(chat_id, False)

    def select_mode(self, chat_id: str, mode: AgentMode) -> AgentMode:
        """Persist a draft Agent Mode unless the Chat has already started."""

        if chat_id in self._deleting:
            raise RuntimeError("Chat is being deleted")
        if chat_id not in self._profiles:
            raise KeyError(chat_id)
        with sqlite3.connect(self._database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT agent_mode, mode_locked FROM active_branches WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            if row is None:
                raise KeyError(chat_id)
            current = AgentMode(row[0])
            if row[1] and current != mode:
                raise ValueError("Agent Mode cannot change after the first Turn")
            if not row[1]:
                connection.execute(
                    "UPDATE active_branches SET agent_mode = ? WHERE chat_id = ?",
                    (mode.value, chat_id),
                )
                current = mode
        self._modes[chat_id] = current
        self._mode_locks[chat_id] = bool(row[1])
        return current

    def lock_mode(self, chat_id: str) -> AgentMode:
        """Atomically lock and return the authoritative Agent Mode."""

        if chat_id in self._deleting:
            raise RuntimeError("Chat is being deleted")
        with sqlite3.connect(self._database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT agent_mode FROM active_branches WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            if row is None:
                raise KeyError(chat_id)
            mode = AgentMode(row[0])
            connection.execute(
                "UPDATE active_branches SET mode_locked = 1 WHERE chat_id = ?",
                (chat_id,),
            )
        self._modes[chat_id] = mode
        self._mode_locks[chat_id] = True
        return mode

    def mark_deleting(self, chat_id: str) -> None:
        self._deleting.add(chat_id)

    async def describe_tool(
        self, chat_id: str, name: str, input_text: str
    ) -> str | None:
        """Generate a compact label without affecting the agent's main run."""

        profile_id = self._profiles.get(chat_id)
        if profile_id is None:
            return None
        profile = self._models[profile_id]
        model = self._title_models.get(profile_id)
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
            self._title_models[profile_id] = model
        payload = safe_text(input_text, max_chars=2000)
        messages = [
            ("system", TOOL_TITLE_SYSTEM_PROMPT),
            (
                "user",
                f"Инструмент: {name}\n<tool-input>\n{payload}\n</tool-input>",
            ),
        ]
        try:
            response = await asyncio.wait_for(
                model.ainvoke(messages),
                timeout=10,
            )
            rejected = message_text(response)
            title = normalize_tool_title(rejected)
            if title is not None:
                return title
            response = await asyncio.wait_for(
                model.ainvoke(
                    [
                        *messages,
                        ("assistant", safe_text(rejected, max_chars=500)),
                        ("user", TOOL_TITLE_RETRY_PROMPT),
                    ]
                ),
                timeout=10,
            )
        except Exception:  # noqa: BLE001 - a cosmetic label must never fail a Turn
            return None
        return normalize_tool_title(message_text(response))

    async def describe_chat(self, chat_id: str, request_text: str) -> str | None:
        """Generate a compact Chat title without affecting the agent's main run."""

        profile_id = self._profiles.get(chat_id)
        if profile_id is None:
            return None
        profile = self._models[profile_id]
        model = self._title_models.get(profile_id)
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
            self._title_models[profile_id] = model
        payload = safe_text(request_text, max_chars=4000)
        try:
            response = await asyncio.wait_for(
                model.ainvoke(
                    [
                        ("system", CHAT_TITLE_SYSTEM_PROMPT),
                        ("user", f"<user-request>\n{payload}\n</user-request>"),
                    ]
                ),
                timeout=10,
            )
        except Exception:  # noqa: BLE001 - a cosmetic title must never fail a Turn
            return None
        return normalize_chat_title(message_text(response))

    async def _graph(self, chat_id: str):
        if chat_id in self._graphs:
            return self._graphs[chat_id]
        if not self.mode_is_locked(chat_id):
            raise RuntimeError("Agent Mode must be locked before building the graph")
        mode = self._modes[chat_id]
        profile = self._models[self._profiles[chat_id]]
        kwargs: dict[str, Any] = {}
        if profile.api_key:
            kwargs["api_key"] = profile.api_key
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        if not profile.streaming:
            kwargs["disable_streaming"] = True
        model = init_chat_model(profile.model, **kwargs)
        backend = await self._sandboxes.backend(chat_id, mode)
        filesystem_tools = (
            READ_ONLY_FILESYSTEM_TOOLS
            if mode is AgentMode.READ_ONLY
            else EXTENDED_FILESYSTEM_TOOLS
        )
        graph = create_deep_agent(
            model=model,
            tools=(
                build_global_memory_tools(self._global_memory, chat_id)
                if self._global_memory is not None
                else []
            ),
            backend=backend,
            checkpointer=await self._checkpointer(),
            system_prompt=agent_system_prompt(mode, self._sandboxes.files_dir(chat_id)),
            middleware=[
                FilesystemMiddleware(
                    backend=backend,
                    tools=list(filesystem_tools),
                )
            ],
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
        if chat_id in self._deleting:
            raise RuntimeError("Chat is being deleted")
        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            if chat_id in self._deleting:
                raise RuntimeError("Chat is being deleted")
            mode = self.lock_mode(chat_id)
            graph = await self._graph(chat_id)
            backend = await self._sandboxes.backend(chat_id, mode)
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
        self.mark_deleting(chat_id)
        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            saver = await self._checkpointer()
            await saver.adelete_thread(chat_id)
            self._profiles.pop(chat_id, None)
            self._modes.pop(chat_id, None)
            self._mode_locks.pop(chat_id, None)
            self._namespaces.pop(chat_id, None)
            self._pending_restore.pop(chat_id, None)
            self._graphs.pop(chat_id, None)
            with sqlite3.connect(self._database) as connection:
                connection.execute(
                    "DELETE FROM active_branches WHERE chat_id = ?", (chat_id,)
                )
