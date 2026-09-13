from __future__ import annotations

import asyncio
import json
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage

from .agent_context import ContextSummary, ModelGuardrails
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
from .agent_memory import AgentMemory
from .chat_bindings import ChatBindings
from .llm_retry import RetryBlock
from .prompts import AGENT_SYSTEM_PROMPT
from .sandbox_files import SandboxFiles
from .sandbox_tools import build_sandbox_tools
from .settings import AgentConfig, LLMRetryConfig, ModelProfile
from .sqlite_history import SQLiteHistory


class AgentExecution:
    """One ReAct loop: load context, call model/tools, persist successful context.

    Context snapshots include summaries. Restoring a Turn is a SQLite state
    replacement, independent of LangGraph's internal checkpoint format.
    """

    def __init__(
        self,
        database: Path,
        models: tuple[ModelProfile, ...],
        sandbox: SandboxFiles,
        chat_bindings: ChatBindings,
        history: SQLiteHistory,
        *,
        config: AgentConfig = AgentConfig(),
        retry_block: RetryBlock | None = None,
    ) -> None:
        self._memory = AgentMemory(database)
        self._models = {model.id: model for model in models}
        self._bindings = chat_bindings
        self._sandbox = sandbox
        self._history = history
        self._config = config
        self._retry = retry_block or RetryBlock(LLMRetryConfig(), init_chat_model)
        self._graphs = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _binding(self, chat_id: str):
        if self._bindings.is_deleting(chat_id):
            raise RuntimeError("Chat is being deleted")
        binding = self._bindings.get(chat_id)
        if binding is None:
            raise KeyError(chat_id)
        return binding

    async def _messages(self, chat_id: str):
        self._binding(chat_id)
        messages = self._memory.load(chat_id)
        if messages is None:
            # Import the visible current history once. Legacy graph state may
            # contain host routes and obsolete tool calls, so it is never run.
            messages = await self._history.context_messages(chat_id)
            self._memory.save(chat_id, messages)
        return messages

    def _graph(self, chat_id: str):
        binding = self._binding(chat_id)
        if chat_id not in self._graphs:
            profile = self._models[binding.profile_id]
            model = self._retry.create_model(
                profile,
                max_tokens=self._config.max_output_tokens,
                disable_streaming=not profile.streaming,
            )
            summary_model = self._retry.create_model(
                profile,
                max_tokens=self._config.summary_tokens,
                disable_streaming=True,
                **profile.summary_options,
            )
            self._graphs[chat_id] = create_agent(
                model=model,
                tools=build_sandbox_tools(self._sandbox.files_dir(chat_id)),
                system_prompt=AGENT_SYSTEM_PROMPT,
                middleware=[
                    ContextSummary(summary_model, self._config, self._retry),
                    ModelGuardrails(self._config, self._retry),
                    ModelCallLimitMiddleware(
                        run_limit=self._config.max_model_calls,
                        exit_behavior="error",
                    ),
                ],
            )
        return self._graphs[chat_id]

    async def checkpoint(self, chat_id: str) -> str:
        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            return self._memory.checkpoint(chat_id, await self._messages(chat_id))

    async def restore(self, chat_id: str, checkpoint: str) -> None:
        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            self._binding(chat_id)
            try:
                token = json.loads(checkpoint)
            except (ValueError, TypeError) as error:
                raise ValueError("Invalid Agent checkpoint") from error
            if not isinstance(token, dict):
                raise ValueError("Invalid Agent checkpoint")
            if token.get("version") == 4:
                self._memory.restore(chat_id, token)
            elif token.get("version") in (None, 2, 3):
                if token.get("chat_id", chat_id) != chat_id:
                    raise ValueError("Agent checkpoint belongs to another Chat")
                messages = await self._history.context_messages(
                    chat_id,
                    before_checkpoint=checkpoint,
                )
                self._memory.save(chat_id, messages)
            else:
                raise ValueError("Unsupported Agent checkpoint version")

    async def run(self, chat_id: str, text: str, emit: EventSink | None = None) -> str:
        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            messages = [*await self._messages(chat_id), HumanMessage(content=text)]
            graph = self._graph(chat_id)
            result = None
            async for event in graph.astream_events(
                {"messages": messages},
                config={"recursion_limit": self._config.max_model_calls * 6 + 20},
                version="v2",
            ):
                kind = event.get("event")
                data = event.get("data", {})
                run_id = str(event.get("run_id", ""))
                if kind == "on_tool_start" and emit:
                    await emit(
                        ToolStarted(
                            id=run_id,
                            name=str(event.get("name") or "tool"),
                            input=safe_text(data.get("input"), max_chars=4000),
                        )
                    )
                elif kind == "on_tool_end" and emit:
                    output = data.get("output")
                    await emit(
                        ToolFinished(
                            id=run_id,
                            output=safe_text(getattr(output, "content", output)),
                        )
                    )
                elif kind == "on_tool_error" and emit:
                    await emit(
                        ToolFailed(id=run_id, error=safe_text(data.get("error")))
                    )
                elif kind == "on_chat_model_stream" and emit:
                    # Internal summary text is never part of the user's answer.
                    if event.get("metadata", {}).get("langgraph_node") != "model":
                        continue
                    chunk = data.get("chunk")
                    if not getattr(chunk, "tool_call_chunks", None):
                        delta = public_text(getattr(chunk, "content", None))
                        if delta:
                            await emit(TextDelta(delta))
                elif kind == "on_chain_end" and not event.get("parent_ids"):
                    result = data.get("output")
            if not isinstance(result, dict) or not result.get("messages"):
                raise RuntimeError("Agent finished without a final state")
            final = result["messages"][-1]
            if not isinstance(final, AIMessage) or final.tool_calls:
                raise RuntimeError("Agent finished without an answer")
            answer = message_text(final)
            if not answer.strip():
                raise RuntimeError(
                    "The model returned an empty answer; history was kept. "
                    "Check the model's reasoning and output token budget."
                )
            self._memory.save(chat_id, result["messages"])
            return answer

    async def delete_chat(self, chat_id: str) -> None:
        self._bindings.mark_deleting(chat_id)
        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            self._memory.delete(chat_id)
            self._bindings.delete(chat_id)
            self._graphs.pop(chat_id, None)

    async def close(self) -> None:
        self._graphs.clear()
