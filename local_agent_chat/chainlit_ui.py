from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import chainlit as cl
from chainlit.context import context_var
from chainlit.utils import utc_now

from .agent_events import AgentEvent, TextDelta, ToolFailed, ToolFinished, ToolStarted
from .tool_logs import format_tool_log


@dataclass(frozen=True)
class ToolDisplay:
    title: str
    icon: str
    show_input: str | bool
    input: str


_SHELL_TOOLS = {"execute", "shell", "bash"}
type ToolTitleResolver = Callable[[str, str], Awaitable[str | None]]


def _input_dict(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _short(value: Any, limit: int = 64) -> str:
    text = str(value or "").strip().splitlines()[0] if value else ""
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _json_input(data: dict[str, Any], fallback: str) -> str:
    if not data:
        return fallback
    return json.dumps(data, ensure_ascii=False, indent=2)


def _shell_input(data: dict[str, Any], fallback: str) -> str:
    command = data.get("command") or data.get("cmd")
    return str(command) if command else fallback


def tool_display(name: str, input_text: str) -> ToolDisplay:
    data = _input_dict(input_text)
    rendered_input = _json_input(data, input_text)
    path = _short(
        data.get("file_path")
        or data.get("path")
        or data.get("pattern")
        or data.get("glob_pattern")
    )
    suffix = f" · {path}" if path else ""

    if name == "read_file":
        return ToolDisplay(f"Чтение файла{suffix}", "file-text", "json", rendered_input)
    if name == "write_file":
        return ToolDisplay(
            f"Создание файла{suffix}", "file-plus-2", "json", rendered_input
        )
    if name == "edit_file":
        return ToolDisplay(
            f"Изменение файла{suffix}", "file-pen-line", "json", rendered_input
        )
    if name == "delete":
        return ToolDisplay(f"Удаление файла{suffix}", "trash-2", "json", rendered_input)
    if name in {"ls", "list_files"}:
        return ToolDisplay(
            f"Список файлов{suffix}", "list-tree", "json", rendered_input
        )
    if name == "search_past_chats":
        query = _short(data.get("query"))
        title = "Поиск в прошлых диалогах" + (f" · {query}" if query else "")
        return ToolDisplay(title, "history", "json", rendered_input)
    if name == "read_past_chat":
        source = _short(data.get("chat_id") or data.get("turn_id"))
        title = "Контекст прошлого диалога" + (f" · {source}" if source else "")
        return ToolDisplay(title, "book-open-text", "json", rendered_input)
    if name in {"glob", "grep", "search"}:
        return ToolDisplay(f"Поиск по файлам{suffix}", "search", "json", rendered_input)
    if name in _SHELL_TOOLS:
        return ToolDisplay(
            "Выполнение системной команды",
            "terminal",
            "bash",
            _shell_input(data, input_text),
        )
    if name in {"task", "subagent"}:
        description = _short(data.get("description") or data.get("prompt"))
        title = "Подзадача" + (f" · {description}" if description else "")
        return ToolDisplay(title, "bot", "json", rendered_input)
    return ToolDisplay(f"Инструмент · {name}", "wrench", "json", rendered_input)


def answer_with_files(answer: str, names: list[str]) -> str:
    content = answer.strip() or "Агент завершил работу без текстового ответа."
    if not names:
        return content
    rendered = ", ".join(f"`{name}`" for name in names)
    return f"{content}\n\n**Изменённые файлы:** {rendered}"


class ChainlitTurnView:
    """Translate one stream of agent events into a compact Chainlit turn."""

    def __init__(
        self,
        *,
        detailed_tools: bool = False,
        tool_title_resolver: ToolTitleResolver | None = None,
    ) -> None:
        self.detailed_tools = detailed_tools
        self.tool_title_resolver = tool_title_resolver
        self.parent_id: str | None = None
        self.root: cl.Step | None = None
        self.answer: cl.Message | None = None
        self.tools: dict[str, cl.Step] = {}
        self.event_sequence = 0
        self.tool_count = 0
        self.terminal = False
        self._tool_title_tasks: set[asyncio.Task[None]] = set()

    async def _update_tool_title(
        self, step: cl.Step, name: str, input_text: str
    ) -> None:
        if self.tool_title_resolver is None:
            return
        try:
            title = await self.tool_title_resolver(name, input_text)
            if title:
                step.name = title
                await step.update()
        except Exception:  # noqa: BLE001 - title generation is cosmetic
            return

    async def finish_tool_titles(self) -> None:
        pending = list(self._tool_title_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _cancel_tool_titles(self) -> None:
        pending = list(self._tool_title_tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def _next_sequence(self) -> int:
        sequence = self.event_sequence
        self.event_sequence += 1
        return sequence

    async def _start_text_segment(self) -> cl.Message:
        message = cl.Message(
            content="",
            parent_id=self.parent_id,
            metadata={
                "event_kind": "assistant_text",
                "event_sequence": self._next_sequence(),
            },
        )
        await message.send()
        self.answer = message
        return message

    async def _finish_text_segment(self) -> None:
        if self.answer is None:
            return
        await self.answer.update()
        self.answer = None

    async def start(self) -> None:
        context = context_var.get(None)
        current_step = context.current_step if context is not None else None
        self.parent_id = current_step.id if current_step is not None else None
        self.root = cl.Step(
            name="Выполнение",
            type="run",
            parent_id=self.parent_id,
            show_input=False,
            default_open=True,
            auto_collapse=True,
            icon="loader-circle",
        )
        self.root.start = utc_now()
        self.root.output = "Агент выполняет запрос…"
        await self.root.send()

    async def handle(self, event: AgentEvent) -> None:
        if self.root is None:
            raise RuntimeError("Turn view must be started before handling events")
        if isinstance(event, TextDelta):
            message = self.answer or await self._start_text_segment()
            await message.stream_token(event.text)
            return
        if isinstance(event, ToolStarted):
            await self._finish_text_segment()
            display = tool_display(event.name, event.input)
            step = cl.Step(
                id=event.id,
                name=display.title,
                type="tool",
                parent_id=self.parent_id,
                show_input=display.show_input,
                default_open=False,
                auto_collapse=False,
                icon=display.icon,
                metadata={
                    "event_kind": "tool_call",
                    "event_sequence": self._next_sequence(),
                    "tool_log_format": 1,
                    "tool_name": event.name,
                },
            )
            step.start = utc_now()
            step.input = display.input
            await step.send()
            self.tools[event.id] = step
            self.tool_count += 1
            if event.name in _SHELL_TOOLS and self.tool_title_resolver is not None:
                task = asyncio.create_task(
                    self._update_tool_title(step, event.name, event.input)
                )
                self._tool_title_tasks.add(task)
                task.add_done_callback(self._tool_title_tasks.discard)
            return
        step = self.tools.get(event.id)
        if step is None:
            return
        step.end = utc_now()
        if isinstance(event, ToolFinished):
            limit = 6000 if self.detailed_tools else 2400
            step.output = format_tool_log(event.output, limit=limit)
            await step.update()
        elif isinstance(event, ToolFailed):
            step.output = format_tool_log(
                event.error or "Инструмент завершился с ошибкой.",
                limit=4000,
            )
            step.is_error = True
            step.default_open = True
            step.auto_collapse = False
            await step.update()

    async def complete(
        self, answer: str, *, elements: list[Any], file_names: list[str]
    ) -> None:
        if self.root is None:
            raise RuntimeError("Turn view was not started")
        if self.terminal:
            return
        self.terminal = True
        await self.finish_tool_titles()
        for step in self.tools.values():
            if step.end is None:
                step.end = utc_now()
                step.output = step.output or format_tool_log("Завершено", limit=4000)
                await step.update()
        self.root.end = utc_now()
        self.root.output = (
            f"Готово · операций: {self.tool_count}" if self.tool_count else "Готово"
        )
        await self.root.update()
        rendered_answer = answer_with_files(answer, file_names)
        if self.answer is None:
            self.answer = cl.Message(
                content=rendered_answer,
                parent_id=self.parent_id,
                metadata={
                    "event_kind": "assistant_final",
                    "event_sequence": self._next_sequence(),
                },
            )
            self.answer.elements = elements
            await self.answer.send()
        else:
            self.answer.content = rendered_answer
            self.answer.metadata["event_kind"] = "assistant_final"
            self.answer.elements = elements
            await self.answer.update()

    async def fail(self, error: str) -> None:
        if self.root is None:
            return
        if self.terminal:
            return
        self.terminal = True
        await self.finish_tool_titles()
        for step in self.tools.values():
            if step.end is None:
                step.end = utc_now()
                step.output = format_tool_log("Выполнение прервано.", limit=4000)
                step.is_error = True
                step.default_open = True
                step.auto_collapse = False
                await step.update()
        self.root.end = utc_now()
        self.root.output = error or "Не удалось завершить запрос."
        self.root.is_error = True
        self.root.default_open = True
        self.root.auto_collapse = False
        await self.root.update()
        if self.answer is None:
            self.answer = cl.Message(
                content=(
                    "Не удалось завершить запрос. "
                    "Подробности показаны в ходе выполнения."
                ),
                parent_id=self.parent_id,
                metadata={
                    "event_kind": "assistant_error",
                    "event_sequence": self._next_sequence(),
                },
            )
            self.answer.is_error = True
            await self.answer.send()
            return
        if not self.answer.content:
            self.answer.content = (
                "Не удалось завершить запрос. Подробности показаны в ходе выполнения."
            )
        self.answer.metadata["event_kind"] = "assistant_error"
        self.answer.is_error = True
        await self.answer.update()

    async def cancel(self) -> None:
        if self.root is None:
            return
        if self.terminal:
            return
        self.terminal = True
        await self._cancel_tool_titles()
        for step in self.tools.values():
            if step.end is None:
                step.end = utc_now()
                step.output = format_tool_log("Остановлено пользователем.", limit=4000)
                step.auto_collapse = False
                await step.update()
        self.root.end = utc_now()
        self.root.output = "Остановлено пользователем"
        self.root.auto_collapse = False
        await self.root.update()
        suffix = "\n\n_Выполнение остановлено пользователем._"
        if self.answer is None:
            self.answer = cl.Message(
                content=suffix.lstrip(),
                parent_id=self.parent_id,
                metadata={
                    "event_kind": "assistant_cancelled",
                    "event_sequence": self._next_sequence(),
                },
            )
            await self.answer.send()
            return
        self.answer.content = (self.answer.content.rstrip() + suffix).lstrip()
        self.answer.metadata["event_kind"] = "assistant_cancelled"
        await self.answer.update()
