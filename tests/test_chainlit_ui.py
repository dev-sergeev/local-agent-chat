import asyncio
import sqlite3
from itertools import count
from pathlib import Path

import chainlit.data as chainlit_data_runtime
import pytest
from chainlit import Step
from chainlit.context import init_http_context
from chainlit.types import Pagination, ThreadFilter
from chainlit.user import User

from local_agent_chat import chainlit_ui
from local_agent_chat.agent_events import (
    TextDelta,
    ToolFailed,
    ToolFinished,
    ToolStarted,
)
from local_agent_chat.chainlit_data import create_chainlit_data_layer
from local_agent_chat.chainlit_ui import (
    ChainlitTurnView,
    tool_display,
)


def test_tool_display_uses_human_labels_and_short_context() -> None:
    listing = tool_display("ls", '{"path": "/"}')
    read = tool_display("read_file", '{"file_path": "src/agent.py"}')

    assert listing.title == "Список файлов · /"
    assert listing.icon == "list-tree"
    assert listing.show_input == "json"
    assert listing.input == '{\n  "path": "/"\n}'
    assert read.title == "Чтение файла · src/agent.py"
    assert read.icon == "file-text"
    assert read.input == '{\n  "file_path": "src/agent.py"\n}'


def test_tool_display_keeps_long_file_paths_distinguishable() -> None:
    common = (
        "/home/jovyan/work/web-ui/.local-agent-chat/sandboxes/"
        "c7dd89f4-d728-4c49-a324-948b91c3dbe5/files"
    )

    readme = tool_display("read_file", f'{{"file_path": "{common}/README.md"}}')
    empty = tool_display(
        "read_file", f'{{"file_path": "{common}/.ui-empty-upload.py"}}'
    )

    assert readme.title.endswith("/README.md")
    assert empty.title.endswith("/.ui-empty-upload.py")
    assert readme.title != empty.title


def test_tool_display_names_global_memory_actions() -> None:
    search = tool_display(
        "search_past_chats", '{"query":"решение по глобальной памяти"}'
    )
    read = tool_display("read_past_chat", '{"chat_id":"chat-42","turn_id":"turn-7"}')

    assert search.title == ("Поиск в прошлых диалогах · решение по глобальной памяти")
    assert search.icon == "history"
    assert read.title == "Контекст прошлого диалога · chat-42"
    assert read.icon == "book-open-text"


def test_tool_display_names_long_term_memory_actions_without_showing_a_fact() -> None:
    remember = tool_display(
        "remember_context", '{"key":"user.name","fact":"[redacted]"}'
    )
    forget = tool_display("forget_context", '{"key":"user.name"}')

    assert remember.title == "Сохранение в память · user.name"
    assert remember.icon == "brain"
    assert "Анна" not in remember.input
    assert forget.title == "Удаление из памяти · user.name"
    assert forget.icon == "eraser"


@pytest.mark.asyncio
async def test_complete_renders_answer_without_streamed_text(monkeypatch) -> None:
    sent = []

    class FakeRoot:
        end = None
        output = ""

        async def update(self):
            return None

    class FakeMessage:
        def __init__(self, content="", **kwargs):
            self.content = content
            self.metadata = kwargs.get("metadata", {})
            self.elements = []

        async def send(self):
            sent.append(self)

    monkeypatch.setattr(chainlit_ui.cl, "Message", FakeMessage)
    monkeypatch.setattr(chainlit_ui, "utc_now", lambda: "now")
    view = ChainlitTurnView()
    view.root = FakeRoot()

    await view.complete("non-streamed answer")

    assert [message.content for message in sent] == ["non-streamed answer"]
    assert sent[0].metadata["event_kind"] == "assistant_final"


@pytest.mark.asyncio
async def test_tool_step_is_collapsed_from_its_first_render(monkeypatch) -> None:
    initial_state = {}

    class FakeStep:
        def __init__(self, **kwargs):
            self.id = kwargs["id"]
            self.default_open = kwargs.get("default_open", False)
            self.auto_collapse = kwargs.get("auto_collapse", False)
            self.start = None
            self.input = ""

        async def send(self):
            initial_state["default_open"] = self.default_open
            initial_state["auto_collapse"] = self.auto_collapse

    monkeypatch.setattr(chainlit_ui.cl, "Step", FakeStep)
    monkeypatch.setattr(chainlit_ui, "utc_now", lambda: "now")
    view = ChainlitTurnView()
    view.root = object()

    await view.handle(ToolStarted("root", "ls", '{"path":"/"}'))

    assert initial_state == {"default_open": False, "auto_collapse": False}


@pytest.mark.asyncio
async def test_turn_view_persists_text_and_tools_in_event_order(monkeypatch) -> None:
    sent = []
    ids = count(1)

    class FakeStep:
        def __init__(
            self, *, id=None, name="", type="undefined", parent_id=None, **kwargs
        ):
            self.id = id or f"step-{next(ids)}"
            self.name = name
            self.type = type
            self.parent_id = parent_id
            self.input = ""
            self.output = ""
            self.start = None
            self.end = None
            self.is_error = False
            self.default_open = kwargs.get("default_open", False)
            self.auto_collapse = kwargs.get("auto_collapse", False)

        async def send(self):
            sent.append(self)

        async def update(self):
            return None

    class FakeMessage:
        def __init__(self, content="", **kwargs):
            self.id = f"message-{next(ids)}"
            self.type = "assistant_message"
            self.parent_id = kwargs.get("parent_id")
            self.metadata = kwargs.get("metadata", {})
            self.content = content
            self.elements = []
            self.is_error = False

        async def send(self):
            sent.append(self)

        async def stream_token(self, token):
            self.content += token

        async def update(self):
            return None

    monkeypatch.setattr(chainlit_ui.cl, "Step", FakeStep)
    monkeypatch.setattr(chainlit_ui.cl, "Message", FakeMessage)
    monkeypatch.setattr(chainlit_ui, "utc_now", lambda: "now")

    view = ChainlitTurnView()
    await view.start()
    await view.handle(TextDelta("Проверю каталог."))
    await view.handle(ToolStarted("root", "ls", '{"path":"/"}'))
    await view.handle(
        ToolFinished(
            "root",
            "\x1b[31mtotal 2\x1b[0m\n---\nfile\n```nested```",
        )
    )
    await view.handle(TextDelta("Проверю ожидаемый путь."))
    await view.handle(ToolStarted("bad", "read_file", '{"file_path":"missing"}'))
    await view.handle(ToolFailed("bad", "path_not_found"))
    await view.handle(TextDelta("Путь не найден, посмотрю корень."))
    await view.handle(ToolStarted("scan", "glob", '{"pattern":"**/*"}'))
    await view.handle(ToolFinished("scan", f"start\n{'x' * 3000}\nend"))
    await view.handle(TextDelta("Готово."))
    await view.complete("Готово.")

    timeline = [
        (item.type, item.id, getattr(item, "content", None))
        for item in sent
        if item.type != "run"
    ]
    assert timeline == [
        ("assistant_message", "message-2", "Проверю каталог."),
        ("tool", "root", None),
        ("assistant_message", "message-3", "Проверю ожидаемый путь."),
        ("tool", "bad", None),
        ("assistant_message", "message-4", "Путь не найден, посмотрю корень."),
        ("tool", "scan", None),
        ("assistant_message", "message-5", "Готово."),
    ]
    assert all(item.parent_id is None for item in sent if item.type == "tool")
    root = next(item for item in sent if item.id == "root")
    assert root.input == '{\n  "path": "/"\n}'
    assert root.output == "````text\ntotal 2\n---\nfile\n```nested```\n````"

    failed = next(item for item in sent if item.id == "bad")
    assert failed.output == "```text\npath_not_found\n```"
    assert failed.is_error is True

    long_log = next(item for item in sent if item.id == "scan").output
    assert long_log.startswith("```text\nstart\n")
    assert "пропущено" in long_log
    assert long_log.endswith("\nend\n```")


@pytest.mark.asyncio
async def test_ordered_event_timeline_survives_chainlit_history_reload(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "chainlit.sqlite3"
    layer = create_chainlit_data_layer(database)
    user = await layer.create_user(User(identifier="local-user", metadata={}))
    assert user is not None
    await layer.update_thread("ordered-chat", user_id=user.id)
    monkeypatch.setattr(chainlit_data_runtime, "_data_layer", layer)
    monkeypatch.setattr(chainlit_data_runtime, "_data_layer_initialized", True)
    init_http_context(thread_id="ordered-chat", user=user)

    async with Step(name="on_message", type="run") as turn:
        view = ChainlitTurnView()
        await view.start()
        await view.handle(TextDelta("Проверю ожидаемый путь."))
        await view.handle(ToolStarted("bad", "read_file", '{"file_path":"missing"}'))
        await view.handle(ToolFailed("bad", "path_not_found"))
        await view.handle(TextDelta("Путь не найден, посмотрю корень."))
        await view.handle(ToolStarted("ls", "ls", '{"path":"/"}'))
        await view.handle(ToolFinished("ls", "total 4"))
        await view.handle(TextDelta("Готово."))
        await view.complete("Готово.")

    for _ in range(100):
        thread = await layer.get_all_user_threads(thread_id="ordered-chat")
        if thread and len(thread[0]["steps"]) == 7:
            stored = {step["id"]: step for step in thread[0]["steps"]}
            if (
                stored["bad"]["output"] == "```text\npath_not_found\n```"
                and stored["ls"]["output"] == "```text\ntotal 4\n```"
                and sorted(
                    step["output"]
                    for step in stored.values()
                    if step["type"] == "assistant_message"
                )
                == sorted(
                    [
                        "Проверю ожидаемый путь.",
                        "Путь не найден, посмотрю корень.",
                        "Готово.",
                    ]
                )
            ):
                break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("ordered Chainlit steps were not persisted")
    await layer.close()

    reopened = create_chainlit_data_layer(database)
    page = await reopened.list_threads(
        Pagination(first=10), ThreadFilter(userId=user.id)
    )
    steps = [step for step in page.data[0]["steps"] if step["type"] != "run"]

    assert [step["metadata"]["event_kind"] for step in steps] == [
        "assistant_text",
        "tool_call",
        "assistant_text",
        "tool_call",
        "assistant_final",
    ]
    assert [step["metadata"]["event_sequence"] for step in steps] == [0, 1, 2, 3, 4]
    assert [step["output"] for step in steps] == [
        "Проверю ожидаемый путь.",
        "```text\npath_not_found\n```",
        "Путь не найден, посмотрю корень.",
        "```text\ntotal 4\n```",
        "Готово.",
    ]
    assert all(step["parentId"] == turn.id for step in steps)
    assert next(step["name"] for step in steps if step["id"] == "ls") == (
        "Список файлов · /"
    )
    await reopened.close()

    with sqlite3.connect(database) as connection:
        flags = {
            row[0]: (bool(row[1]), bool(row[2]))
            for row in connection.execute(
                'SELECT id, "defaultOpen", "autoCollapse" FROM steps '
                'WHERE id IN ("bad", "ls")'
            )
        }
    assert flags == {"bad": (True, False), "ls": (False, False)}
