from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import aiosqlite
from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT, SubAgent
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
from .agent_modes import READ_FILESYSTEM_TOOLS
from .chat_bindings import ChatBinding, ChatBindings
from .llm_retry import RetryBlock
from .long_term_memory import REMEMBER_CONTEXT_TOOL, MarkdownMemory
from .memory_tools import build_global_memory_tools
from .prompts import agent_system_prompt
from .sandbox_provider import LocalSandboxManager
from .settings import LLMRetryConfig, ModelProfile
from .sqlite_history import SQLiteHistory


class DeepAgentExecution:
    """Own Deep Agents graphs, Agent Memory, and event translation."""

    def __init__(
        self,
        database: Path,
        models: tuple[ModelProfile, ...],
        sandboxes: LocalSandboxManager,
        chat_bindings: ChatBindings,
        global_memory: SQLiteHistory | None = None,
        long_term_memory: MarkdownMemory | None = None,
        llm_retry: LLMRetryConfig | None = None,
        retry_block: RetryBlock | None = None,
        skills_dir: Path | None = None,
        recursion_limit: int = 100,
    ) -> None:
        if recursion_limit < 1:
            raise ValueError("Agent graph recursion limit must be positive")
        if retry_block is not None and llm_retry is not None:
            raise ValueError("Pass either retry_block or llm_retry, not both")
        self._database = database
        self._models = {model.id: model for model in models}
        self._bindings = chat_bindings
        self._sandboxes = sandboxes
        self._global_memory = global_memory
        self._long_term_memory = long_term_memory
        self._skill_sources: list[str] | None = None
        if skills_dir is not None:
            resolved_skills = skills_dir.resolve()
            if not resolved_skills.is_dir():
                raise ValueError(
                    f"Agent skills directory does not exist: {resolved_skills}"
                )
            self._skill_sources = [resolved_skills.as_posix()]
        self._retry_block = retry_block or RetryBlock(
            llm_retry or LLMRetryConfig(), init_chat_model
        )
        self._recursion_limit = recursion_limit
        self._connection: aiosqlite.Connection | None = None
        self._saver: AsyncSqliteSaver | None = None
        self._checkpointer_lock = asyncio.Lock()
        self._graphs: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _checkpointer(self) -> AsyncSqliteSaver:
        if self._saver is not None:
            return self._saver
        async with self._checkpointer_lock:
            if self._saver is None:
                connection = await aiosqlite.connect(self._database)
                saver = AsyncSqliteSaver(connection)
                try:
                    await saver.setup()
                except BaseException:
                    await connection.close()
                    raise
                self._connection = connection
                self._saver = saver
        return self._saver

    def _active_binding(self, chat_id: str) -> ChatBinding:
        if self._bindings.is_deleting(chat_id):
            raise RuntimeError("Chat is being deleted")
        binding = self._bindings.get(chat_id)
        if binding is None:
            raise KeyError(chat_id)
        return binding

    async def _graph(self, chat_id: str):
        if chat_id in self._graphs:
            return self._graphs[chat_id]
        binding = self._active_binding(chat_id)
        if not binding.mode_locked:
            raise RuntimeError("Agent Mode must be locked before building the graph")
        mode = binding.mode
        profile = self._models[binding.profile_id]
        model_kwargs: dict[str, Any] = {}
        if not profile.streaming:
            model_kwargs["disable_streaming"] = True
        model = self._retry_block.create_model(profile, **model_kwargs)
        backend = await self._sandboxes.backend(chat_id, mode)
        agent_tools = (
            build_global_memory_tools(self._global_memory, chat_id)
            if self._global_memory is not None
            else []
        )

        def base_middleware() -> list[Any]:
            return [
                FilesystemMiddleware(
                    backend=backend,
                    tools=list(READ_FILESYSTEM_TOOLS),
                ),
                self._retry_block.summarization_middleware(model, backend),
            ]

        middleware = base_middleware()
        subagent_middleware = base_middleware()
        subagent_description = (
            "General-purpose agent for complex research and multi-step tasks."
        )
        if self._long_term_memory is not None:
            middleware.append(self._long_term_memory.agent_middleware())
            subagent_middleware.append(self._long_term_memory.reference_middleware())
            subagent_description += (
                " It receives Long-term Memory as reference context without its "
                "mutation tools; report updates to the calling Agent."
            )
        general_purpose = cast(
            SubAgent,
            {
                **GENERAL_PURPOSE_SUBAGENT,
                "description": subagent_description,
                "middleware": subagent_middleware,
            },
        )
        if self._skill_sources is not None:
            general_purpose["skills"] = self._skill_sources
        graph = create_deep_agent(
            model=model,
            tools=agent_tools,
            backend=backend,
            subagents=[general_purpose],
            skills=self._skill_sources,
            checkpointer=await self._checkpointer(),
            system_prompt=agent_system_prompt(mode, self._sandboxes.files_dir(chat_id)),
            middleware=middleware,
        )
        self._graphs[chat_id] = graph
        return graph

    def _base_config(self, chat_id: str) -> dict[str, Any]:
        binding = self._active_binding(chat_id)
        return {
            "recursion_limit": self._recursion_limit,
            "configurable": {
                "thread_id": binding.memory_thread_id,
                "checkpoint_ns": "",
            },
        }

    async def checkpoint(self, chat_id: str) -> str:
        if self._bindings.is_deleting(chat_id):
            raise RuntimeError("Chat is being deleted")
        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            if self._bindings.is_deleting(chat_id):
                raise RuntimeError("Chat is being deleted")
            return await self._checkpoint_unlocked(chat_id)

    async def _checkpoint_unlocked(self, chat_id: str) -> str:
        saver = await self._checkpointer()
        config = self._base_config(chat_id)
        item = await saver.aget_tuple(config)
        checkpoint_id = (
            item.config["configurable"].get("checkpoint_id") if item else None
        )
        return json.dumps(
            {
                "version": 3,
                "chat_id": chat_id,
                "memory_thread_id": config["configurable"]["thread_id"],
                "checkpoint_ns": config["configurable"]["checkpoint_ns"],
                "checkpoint_id": checkpoint_id,
            }
        )

    async def restore(self, chat_id: str, checkpoint: str) -> None:
        if self._bindings.is_deleting(chat_id):
            raise RuntimeError("Chat is being deleted")
        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            if self._bindings.is_deleting(chat_id):
                raise RuntimeError("Chat is being deleted")
            await self._restore_unlocked(chat_id, checkpoint)

    async def _restore_unlocked(self, chat_id: str, checkpoint: str) -> None:
        self._active_binding(chat_id)
        try:
            token = json.loads(checkpoint)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError("Invalid Agent checkpoint") from error
        if not isinstance(token, dict):
            raise ValueError("Invalid Agent checkpoint")
        version = token.get("version")
        if version not in {None, 2, 3}:
            raise ValueError(f"Unsupported Agent checkpoint version: {version}")
        if version in {2, 3} and token.get("chat_id") != chat_id:
            raise ValueError("Agent checkpoint belongs to another Chat")
        memory_thread_id = token.get("memory_thread_id") if version == 3 else chat_id
        if not isinstance(memory_thread_id, str):
            raise ValueError("Invalid Agent checkpoint")
        checkpoint_ns = token.get("checkpoint_ns")
        checkpoint_id = token.get("checkpoint_id")
        if not isinstance(checkpoint_ns, str) or (
            checkpoint_id is not None and not isinstance(checkpoint_id, str)
        ):
            raise ValueError("Invalid Agent checkpoint")
        if version == 3 and checkpoint_ns:
            raise ValueError("Invalid Agent checkpoint namespace")
        if checkpoint_id:
            if not self._bindings.owns_memory_thread(chat_id, memory_thread_id):
                raise ValueError("Agent Memory thread belongs to another Chat")
            saver = await self._checkpointer()
            source = await saver.aget_tuple(
                {
                    "configurable": {
                        "thread_id": memory_thread_id,
                        "checkpoint_ns": "",
                        "checkpoint_id": checkpoint_id,
                    }
                }
            )
            if source is None:
                raise ValueError("Agent checkpoint does not exist")
            restored_thread_id = self._bindings.reserve_memory_thread(chat_id)
            target_config = await saver.aput(
                {
                    "configurable": {
                        "thread_id": restored_thread_id,
                        "checkpoint_ns": "",
                    }
                },
                source.checkpoint,
                source.metadata,
                source.checkpoint.get("channel_versions", {}),
            )
            pending_by_task: dict[str, list[tuple[str, Any]]] = {}
            for task_id, channel, value in source.pending_writes or []:
                pending_by_task.setdefault(task_id, []).append((channel, value))
            for task_id, writes in pending_by_task.items():
                await saver.aput_writes(target_config, writes, task_id)
            self._bindings.use_memory_thread(chat_id, restored_thread_id)
        else:
            self._bindings.new_memory_thread(chat_id)

    async def run(self, chat_id: str, text: str, emit: EventSink | None = None) -> str:
        if self._bindings.is_deleting(chat_id):
            raise RuntimeError("Chat is being deleted")
        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            if self._bindings.is_deleting(chat_id):
                raise RuntimeError("Chat is being deleted")
            binding = self._active_binding(chat_id)
            if not binding.mode_locked:
                raise RuntimeError("Agent Mode must be locked before execution")
            graph = await self._graph(chat_id)
            config = self._base_config(chat_id)
            config["recursion_limit"] = self._recursion_limit
            result: dict[str, Any] | None = None
            redacted_tool_runs: set[str] = set()
            initial_input = {"messages": [{"role": "user", "content": text}]}
            async for event in graph.astream_events(
                initial_input,
                config=config,
                version="v2",
                durability="sync",
            ):
                event_type = event.get("event")
                data = event.get("data", {})
                run_id = str(event.get("run_id", ""))
                if event_type == "on_tool_start":
                    if emit is not None:
                        tool_name = str(event.get("name") or "tool")
                        tool_input = data.get("input")
                        if tool_name == REMEMBER_CONTEXT_TOOL:
                            redacted_tool_runs.add(run_id)
                            key = (
                                tool_input.get("key")
                                if isinstance(tool_input, dict)
                                else None
                            )
                            tool_input = {"key": key, "fact": "[redacted]"}
                        await emit(
                            ToolStarted(
                                id=run_id,
                                name=tool_name,
                                input=safe_text(tool_input, max_chars=4000),
                            )
                        )
                elif event_type == "on_tool_end":
                    if emit is not None:
                        output = data.get("output")
                        await emit(
                            ToolFinished(
                                id=run_id,
                                output=safe_text(
                                    getattr(output, "content", output),
                                    max_chars=6000,
                                ),
                            )
                        )
                    redacted_tool_runs.discard(run_id)
                elif event_type == "on_tool_error":
                    if emit is not None:
                        error = (
                            "Long-term Memory tool failed"
                            if run_id in redacted_tool_runs
                            or event.get("name") == REMEMBER_CONTEXT_TOOL
                            else safe_text(data.get("error"), max_chars=4000)
                        )
                        await emit(
                            ToolFailed(
                                id=run_id,
                                error=error,
                            )
                        )
                    redacted_tool_runs.discard(run_id)
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
            if result is None:
                raise RuntimeError("Agent finished without a final state")
            message = result["messages"][-1]
            return message_text(message)

    async def close(self) -> None:
        async with self._checkpointer_lock:
            if self._connection is not None:
                await self._connection.close()
                self._connection = None
                self._saver = None

    async def delete_chat(self, chat_id: str) -> None:
        self._bindings.mark_deleting(chat_id)
        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            saver = await self._checkpointer()
            for memory_thread_id in self._bindings.memory_threads(chat_id):
                await saver.adelete_thread(memory_thread_id)
            self._bindings.delete(chat_id)
            self._graphs.pop(chat_id, None)
