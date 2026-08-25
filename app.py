from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Switch
from chainlit.server import app
from fastapi import HTTPException
from fastapi.responses import FileResponse

from local_agent_chat.agent_events import safe_text
from local_agent_chat.agent_modes import AgentMode
from local_agent_chat.agent_service import AgentService
from local_agent_chat.chainlit_data import create_chainlit_data_layer
from local_agent_chat.chainlit_mode_guard import install_mode_acceptance_guard
from local_agent_chat.chainlit_ui import ChainlitTurnView
from local_agent_chat.chat_titles import (
    CHAT_TITLE_FALLBACK,
    CHAT_TITLE_PENDING,
    DEFAULT_CHAT_TITLE,
)
from local_agent_chat.local_storage import LocalStorageClient
from local_agent_chat.proxy_prefix import (
    RestoreProxyMethodMiddleware,
    RestoreProxyPrefixMiddleware,
)
from local_agent_chat.runtime import ChatRuntime
from local_agent_chat.sandbox_files import SandboxFiles
from local_agent_chat.sandbox_provider import LocalSandboxManager
from local_agent_chat.settings import load_settings
from local_agent_chat.sqlite_history import SQLiteHistory

settings = load_settings()
app.add_middleware(RestoreProxyPrefixMiddleware, prefix=settings.root_path)
app.add_middleware(RestoreProxyMethodMiddleware)
settings.data_dir.mkdir(parents=True, exist_ok=True)
storage = LocalStorageClient(settings.data_dir / "blobs", f"{settings.root_path}/files")
chainlit_layer = create_chainlit_data_layer(
    settings.data_dir / "chainlit.sqlite3", storage
)
sandbox_files = SandboxFiles(
    settings.data_dir / "sandboxes",
    max_file_bytes=100 * 1024 * 1024,
    max_chat_bytes=1024 * 1024 * 1024,
)
sandbox_manager = LocalSandboxManager(sandbox_files)
runtime_history = SQLiteHistory(settings.data_dir / "runtime-history.sqlite3")
agent_service = AgentService(
    settings.data_dir / "checkpoints.sqlite3",
    settings.models,
    sandbox_manager,
    global_memory=runtime_history,
)


def _lock_mode_at_message_acceptance(
    chat_id: str, selected_profile: str | None
) -> None:
    profile_id = (
        agent_service.profile_for(chat_id) or selected_profile or settings.models[0].id
    )
    agent_service.set_profile(chat_id, profile_id)
    agent_service.lock_mode(chat_id)


def _select_mode_at_settings_acceptance(
    chat_id: str, selected_profile: str | None, extended: bool
) -> None:
    profile_id = (
        agent_service.profile_for(chat_id) or selected_profile or settings.models[0].id
    )
    agent_service.set_profile(chat_id, profile_id)
    requested = AgentMode.EXTENDED if extended else AgentMode.READ_ONLY
    try:
        agent_service.select_mode(chat_id, requested)
    except ValueError:
        pass


install_mode_acceptance_guard(
    _select_mode_at_settings_acceptance,
    _lock_mode_at_message_acceptance,
)

runtime = ChatRuntime(
    agent=agent_service,
    sandbox=sandbox_files,
    history=runtime_history,
)
active_views: dict[str, ChainlitTurnView] = {}
chat_title_tasks: dict[str, asyncio.Task[None]] = {}
chat_turn_locks: dict[str, asyncio.Lock] = {}
deleting_chats: set[str] = set()


def _title_task_finished(chat_id: str, task: asyncio.Task[None]) -> None:
    if chat_title_tasks.get(chat_id) is task:
        chat_title_tasks.pop(chat_id, None)
    if not task.cancelled():
        task.exception()


def _start_chat_title(chat_id: str, request_text: str) -> None:
    current = chat_title_tasks.get(chat_id)
    if current is not None and not current.done():
        return
    task = asyncio.create_task(_publish_chat_title(chat_id, request_text))
    chat_title_tasks[chat_id] = task
    task.add_done_callback(lambda finished: _title_task_finished(chat_id, finished))


async def cleanup_chat(chat_id: str) -> None:
    deleting_chats.add(chat_id)
    agent_service.mark_deleting(chat_id)
    title_task = chat_title_tasks.pop(chat_id, None)
    if title_task is not None:
        title_task.cancel()
        await asyncio.gather(title_task, return_exceptions=True)
    async with chat_turn_locks.setdefault(chat_id, asyncio.Lock()):

        async def delete_state() -> None:
            await agent_service.delete_chat(chat_id)
            await sandbox_manager.delete_chat(chat_id)
            await sandbox_files.delete_chat(chat_id)
            await runtime_history.delete_chat(chat_id)

        await runtime.delete_chat(chat_id, delete_state)


chainlit_layer.chat_cleanup = cleanup_chat


@cl.data_layer
def data_layer():
    return chainlit_layer


@cl.header_auth_callback
async def local_user(_headers):
    return cl.User(identifier="local-user", metadata={"role": "local"})


@cl.set_chat_profiles
async def chat_profiles(_user):
    starters = [
        cl.Starter(
            label="Изучить файлы",
            message="Изучи файлы в песочнице и кратко опиши структуру и назначение проекта.",
        ),
        cl.Starter(
            label="Исправить проблему",
            message="Найди причину проблемы в приложенных файлах, исправь её и проверь результат тестами.",
        ),
        cl.Starter(
            label="Обработать данные",
            message="Изучи приложенные файлы, обработай данные и сохрани результат в новом файле.",
        ),
        cl.Starter(
            label="Объяснить код",
            message="Проанализируй приложенный код и объясни, как он работает и что можно улучшить.",
        ),
    ]
    return [
        cl.ChatProfile(
            name=model.id,
            display_name=model.label,
            markdown_description=f"Model Profile: `{model.model}`",
            default=index == 0,
            starters=starters,
        )
        for index, model in enumerate(settings.models)
    ]


def _thread_id() -> str:
    return cl.context.session.thread_id


def _selected_profile(chat_id: str) -> str:
    return (
        agent_service.profile_for(chat_id)
        or cl.user_session.get("chat_profile")
        or settings.models[0].id
    )


async def _publish_chat_title(chat_id: str, request_text: str) -> None:
    """Persist and publish a semantic title without affecting the Turn."""

    try:
        await chainlit_layer.wait_for_initial_name(chat_id)
        if not await chainlit_layer.begin_chat_title(chat_id):
            return
        await cl.context.emitter.emit(
            "first_interaction",
            {"interaction": DEFAULT_CHAT_TITLE, "thread_id": chat_id},
        )
        title = await agent_service.describe_chat(chat_id, request_text)
        rendered = title or DEFAULT_CHAT_TITLE
        applied = await chainlit_layer.complete_chat_title(
            chat_id, rendered, fallback=title is None
        )
        if applied and title:
            await cl.context.emitter.emit(
                "first_interaction",
                {"interaction": title, "thread_id": chat_id},
            )
    except Exception:  # noqa: BLE001 - a cosmetic title must never fail a Turn
        return


async def _send_chat_settings(*, refresh: bool = False) -> None:
    chat_id = _thread_id()
    mode = agent_service.mode_for(chat_id) or AgentMode.READ_ONLY
    detailed = bool(cl.user_session.get("show_tool_details", False))
    chat_settings = cl.ChatSettings(
        [
            Switch(
                id="extended_mode",
                label="Расширенный режим",
                initial=mode is AgentMode.EXTENDED,
                tooltip=(
                    "Добавляет создание, изменение и удаление файлов, а также "
                    "выполнение команд. Выбор фиксируется после первого сообщения."
                ),
                description=(
                    "Выключено: чтение и поиск по диску. Включено: полный "
                    "доступ с правами процесса приложения."
                ),
                disabled=agent_service.mode_is_locked(chat_id),
            ),
            Switch(
                id="show_tool_details",
                label="Подробные результаты инструментов",
                initial=detailed,
                tooltip="Показывать больше stdout, stderr и результатов файловых операций.",
            ),
        ]
    )
    if refresh:
        await chat_settings.refresh()
    else:
        await chat_settings.send()


async def _recover_mode_lock(chat_id: str, profile_id: str) -> AgentMode:
    """Lock the selected mode if Chainlit persisted a request before its handler."""

    mode = agent_service.mode_for(chat_id) or AgentMode.READ_ONLY
    if agent_service.mode_is_locked(chat_id):
        return mode
    if not await chainlit_layer.has_user_request(chat_id):
        return mode
    mode = agent_service.lock_mode(chat_id)
    await chainlit_layer.update_thread(
        chat_id,
        metadata={
            "model_profile": profile_id,
            "agent_mode": mode.value,
            "agent_mode_locked": True,
        },
    )
    return mode


@cl.on_chat_start
async def on_chat_start():
    chat_id = _thread_id()
    profile_id = _selected_profile(chat_id)
    agent_service.set_profile(chat_id, profile_id)
    cl.user_session.set("model_profile", profile_id)
    cl.user_session.set("agent_mode", agent_service.mode_for(chat_id).value)
    await _send_chat_settings()


@cl.on_chat_resume
async def on_chat_resume(thread):
    chat_id = thread["id"]
    profile_id = (
        agent_service.profile_for(chat_id)
        or thread.get("metadata", {}).get("model_profile")
        or settings.models[0].id
    )
    agent_service.set_profile(chat_id, profile_id)
    mode = await _recover_mode_lock(chat_id, profile_id)
    cl.user_session.set("model_profile", profile_id)
    cl.user_session.set("agent_mode", mode.value)
    if await chainlit_layer.chat_title_state(chat_id) in {
        CHAT_TITLE_PENDING,
        CHAT_TITLE_FALLBACK,
    }:
        request_text = await chainlit_layer.first_user_request(chat_id)
        if request_text:
            _start_chat_title(chat_id, request_text)
    await _send_chat_settings()


@cl.on_settings_update
async def on_settings_update(updated):
    cl.user_session.set(
        "show_tool_details", bool(updated.get("show_tool_details", False))
    )
    requested_mode = (
        AgentMode.EXTENDED
        if updated.get("extended_mode") is True
        else AgentMode.READ_ONLY
    )
    chat_id = _thread_id()
    profile_id = _selected_profile(chat_id)
    mode = await _recover_mode_lock(chat_id, profile_id)
    if agent_service.mode_is_locked(chat_id):
        cl.user_session.set("agent_mode", mode.value)
        await _send_chat_settings(refresh=True)
        return
    try:
        mode = agent_service.select_mode(chat_id, requested_mode)
    except ValueError:
        mode = agent_service.mode_for(chat_id) or AgentMode.READ_ONLY
    cl.user_session.set("agent_mode", mode.value)
    if agent_service.mode_is_locked(chat_id):
        await _send_chat_settings(refresh=True)


def _changed_file_elements(chat_id: str, names: list[str]):
    elements = []
    for name in names:
        path = str(sandbox_files.files_dir(chat_id) / name)
        mime, _ = mimetypes.guess_type(name)
        if mime and mime.startswith("image/"):
            elements.append(cl.Image(name=name, path=path, display="inline"))
        elif mime == "application/pdf":
            elements.append(cl.Pdf(name=name, path=path, display="side"))
        else:
            elements.append(cl.File(name=name, path=path, display="side", mime=mime))
    return elements


async def _handle_message(message: cl.Message, chat_id: str) -> None:
    profile_id = _selected_profile(chat_id)
    agent_service.set_profile(chat_id, profile_id)
    mode = agent_service.lock_mode(chat_id)
    cl.user_session.set("agent_mode", mode.value)
    await _send_chat_settings(refresh=True)
    await chainlit_layer.update_thread(
        chat_id,
        metadata={
            "model_profile": profile_id,
            "agent_mode": mode.value,
            "agent_mode_locked": True,
        },
    )

    is_revision = await runtime.has_turn(message.id)
    is_first_turn = not is_revision and not await runtime_history.has_chat(chat_id)
    title_state = await chainlit_layer.chat_title_state(chat_id)
    should_title = False
    if is_first_turn:
        should_title = await chainlit_layer.begin_chat_title(chat_id)
    elif title_state in {CHAT_TITLE_PENDING, CHAT_TITLE_FALLBACK}:
        should_title = await chainlit_layer.begin_chat_title(chat_id)
    if should_title:
        title_source = (
            message.content
            if is_first_turn
            else await chainlit_layer.first_user_request(chat_id) or message.content
        )
        _start_chat_title(chat_id, title_source)

    for element in message.elements or []:
        source = getattr(element, "path", None)
        if source:
            await sandbox_files.upload(
                chat_id, Path(source), element.name or Path(source).name
            )

    before_files = sandbox_files.manifest(chat_id)
    if is_revision:
        await chainlit_layer.wait_for_revision(message.id)
        await chainlit_layer.truncate_revision(message.id)
    view = ChainlitTurnView(
        detailed_tools=bool(cl.user_session.get("show_tool_details", False)),
        tool_title_resolver=lambda name, input_text: agent_service.describe_tool(
            chat_id, name, input_text
        ),
    )
    active_views[chat_id] = view
    await view.start()
    try:
        if is_revision:
            answer = await runtime.revise(
                chat_id, message.id, message.content, view.handle
            )
        else:
            answer = await runtime.submit(
                chat_id, message.id, message.content, view.handle
            )
    except asyncio.CancelledError:
        await asyncio.shield(view.cancel())
        if is_revision:
            await asyncio.shield(chainlit_layer.restore_revision(message.id))
        raise
    except Exception as error:  # noqa: BLE001 - provider/tool failures end the Turn
        await view.fail(safe_text(error, max_chars=2000))
        if is_revision:
            await chainlit_layer.restore_revision(message.id)
        return
    finally:
        active_views.pop(chat_id, None)

    after_files = sandbox_files.manifest(chat_id)
    changed_names = [
        name for name, digest in after_files.items() if before_files.get(name) != digest
    ]
    await view.complete(
        answer,
        elements=_changed_file_elements(chat_id, changed_names),
        file_names=changed_names,
    )
    if is_revision:
        await chainlit_layer.commit_revision(message.id)


@cl.on_message
async def on_message(message: cl.Message):
    chat_id = _thread_id()
    if chat_id in deleting_chats:
        return
    async with chat_turn_locks.setdefault(chat_id, asyncio.Lock()):
        if chat_id in deleting_chats:
            return
        await _handle_message(message, chat_id)


@cl.on_stop
async def on_stop():
    chat_id = _thread_id()
    profile_id = agent_service.profile_for(chat_id)
    if profile_id is not None and chat_id not in deleting_chats:
        mode = await _recover_mode_lock(chat_id, profile_id)
        cl.user_session.set("agent_mode", mode.value)
        if agent_service.mode_is_locked(chat_id):
            await _send_chat_settings(refresh=True)
    view = active_views.get(chat_id)
    if view is not None:
        try:
            await view.cancel()
        except RuntimeError:
            pass


@app.get("/files/{object_key:path}", include_in_schema=False)
async def local_file(object_key: str):
    try:
        path = storage.path_for(object_key)
    except ValueError as error:
        raise HTTPException(status_code=404) from error
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(
        path, media_type=storage.media_type(object_key), filename=path.name
    )


@app.on_event("shutdown")
async def close_resources() -> None:
    pending_titles = list(chat_title_tasks.values())
    for task in pending_titles:
        task.cancel()
    if pending_titles:
        await asyncio.gather(*pending_titles, return_exceptions=True)
    await agent_service.close()
