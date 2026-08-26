from __future__ import annotations

import asyncio
import mimetypes
from collections.abc import Awaitable
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Switch
from chainlit.server import app, router
from fastapi import HTTPException
from fastapi.responses import FileResponse
from langchain.chat_models import init_chat_model

from local_agent_chat.agent_events import safe_text
from local_agent_chat.agent_modes import AgentMode
from local_agent_chat.auxiliary_labels import AuxiliaryLabels
from local_agent_chat.chainlit_data import create_chainlit_data_layer
from local_agent_chat.chainlit_mode_guard import install_mode_acceptance_guard
from local_agent_chat.chainlit_stop import install_localized_stop_compatibility
from local_agent_chat.chainlit_ui import ChainlitTurnView
from local_agent_chat.chainlit_uploads import (
    install_empty_file_upload_compatibility,
    install_unrestricted_file_upload_compatibility,
)
from local_agent_chat.chat_bindings import ChatBinding, ChatBindings
from local_agent_chat.chat_configuration import ChatConfigurations
from local_agent_chat.chat_titles import (
    CHAT_TITLE_FALLBACK,
    CHAT_TITLE_PENDING,
    DEFAULT_CHAT_TITLE,
)
from local_agent_chat.deep_agent_execution import DeepAgentExecution
from local_agent_chat.llm_retry import RetryBlock
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
install_empty_file_upload_compatibility()
install_unrestricted_file_upload_compatibility()
install_localized_stop_compatibility()
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
checkpoint_database = settings.data_dir / "checkpoints.sqlite3"
chat_bindings = ChatBindings(
    checkpoint_database,
    (model.id for model in settings.models),
)
retry_block = RetryBlock(settings.llm_retry, init_chat_model)
auxiliary_labels = AuxiliaryLabels(settings.models, chat_bindings, retry_block)
agent_execution = DeepAgentExecution(
    checkpoint_database,
    settings.models,
    sandbox_manager,
    global_memory=runtime_history,
    retry_block=retry_block,
    skills_dir=Path(__file__).resolve().parent / "skills",
    chat_bindings=chat_bindings,
)
chat_configurations = ChatConfigurations(
    chat_bindings,
    (model.id for model in settings.models),
    chainlit_layer.has_user_request,
)
install_mode_acceptance_guard(
    lambda chat_id, profile, extended: chat_configurations.select_mode(
        chat_id,
        AgentMode.EXTENDED if extended else AgentMode.READ_ONLY,
        profile,
    ),
    chat_configurations.accept_message,
)

runtime = ChatRuntime(
    agent=agent_execution,
    sandbox=sandbox_files,
    history=runtime_history,
)
active_views: dict[str, ChainlitTurnView] = {}
chat_title_tasks: dict[str, asyncio.Task[None]] = {}
chat_turn_locks: dict[str, asyncio.Lock] = {}
deleting_chats: set[str] = set()


class _TurnRunFailed(Exception):
    """The Turn failure is already rendered for the user."""


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
    chat_bindings.mark_deleting(chat_id)
    title_task = chat_title_tasks.pop(chat_id, None)
    if title_task is not None:
        title_task.cancel()
        await asyncio.gather(title_task, return_exceptions=True)
    async with chat_turn_locks.setdefault(chat_id, asyncio.Lock()):

        async def delete_state() -> None:
            await agent_execution.delete_chat(chat_id)
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
        title = await auxiliary_labels.describe_chat(chat_id, request_text)
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


def _remember_chat_configuration(binding: ChatBinding) -> None:
    cl.user_session.set("model_profile", binding.profile_id)
    cl.user_session.set("agent_mode", binding.mode.value)


async def _persist_chat_configuration(chat_id: str, binding: ChatBinding) -> None:
    await chainlit_layer.update_thread(
        chat_id,
        metadata={
            "model_profile": binding.profile_id,
            "agent_mode": binding.mode.value,
            "agent_mode_locked": binding.mode_locked,
        },
    )


async def _sync_chat_configuration(
    chat_id: str, binding: ChatBinding, *, refresh: bool
) -> None:
    _remember_chat_configuration(binding)
    await _persist_chat_configuration(chat_id, binding)
    await _send_chat_settings(binding, refresh=refresh)


async def _send_chat_settings(binding: ChatBinding, *, refresh: bool = False) -> None:
    detailed = bool(cl.user_session.get("show_tool_details", False))
    chat_settings = cl.ChatSettings(
        [
            Switch(
                id="extended_mode",
                label="Расширенный режим",
                initial=binding.mode is AgentMode.EXTENDED,
                tooltip=(
                    "Добавляет создание, изменение и удаление файлов, а также "
                    "выполнение команд. Выбор фиксируется после первого сообщения."
                ),
                description=(
                    "Выключено: чтение и поиск по диску. Включено: полный "
                    "доступ с правами процесса приложения."
                ),
                disabled=binding.mode_locked,
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


@cl.on_chat_start
async def on_chat_start():
    chat_id = _thread_id()
    binding = chat_configurations.open(
        chat_id,
        cl.user_session.get("chat_profile"),
    )
    _remember_chat_configuration(binding)
    await _send_chat_settings(binding)


@cl.on_chat_resume
async def on_chat_resume(thread):
    chat_id = thread["id"]
    binding = await chat_configurations.recover(
        chat_id,
        thread.get("metadata", {}).get("model_profile"),
        cl.user_session.get("chat_profile"),
    )
    if await chainlit_layer.chat_title_state(chat_id) in {
        CHAT_TITLE_PENDING,
        CHAT_TITLE_FALLBACK,
    }:
        request_text = await chainlit_layer.first_user_request(chat_id)
        if request_text:
            _start_chat_title(chat_id, request_text)
    await _sync_chat_configuration(chat_id, binding, refresh=False)


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
    profile_hint = cl.user_session.get("chat_profile")
    binding = await chat_configurations.recover(chat_id, profile_hint)
    if binding.mode_locked:
        await _sync_chat_configuration(chat_id, binding, refresh=True)
        return
    binding = chat_configurations.select_mode(
        chat_id,
        requested_mode,
        profile_hint,
    )
    if binding.mode_locked:
        await _sync_chat_configuration(chat_id, binding, refresh=True)
    else:
        _remember_chat_configuration(binding)


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


async def _run_turn(view: ChainlitTurnView, operation: Awaitable[str]) -> str:
    try:
        return await operation
    except asyncio.CancelledError:
        await asyncio.shield(view.cancel())
        raise
    except Exception as error:  # noqa: BLE001 - provider/tool failures end the Turn
        await view.fail(safe_text(error, max_chars=2000))
        raise _TurnRunFailed from error


async def _complete_turn(
    view: ChainlitTurnView,
    chat_id: str,
    before_files: dict[str, str],
    answer: str,
) -> None:
    after_files = await asyncio.to_thread(sandbox_files.manifest, chat_id)
    changed_names = [
        name for name, digest in after_files.items() if before_files.get(name) != digest
    ]
    await view.complete(
        answer,
        elements=_changed_file_elements(chat_id, changed_names),
        file_names=changed_names,
    )


async def _handle_message(message: cl.Message, chat_id: str) -> None:
    binding = chat_configurations.accept_message(
        chat_id,
        cl.user_session.get("chat_profile"),
    )
    await _sync_chat_configuration(chat_id, binding, refresh=True)

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

    before_files = await asyncio.to_thread(sandbox_files.manifest, chat_id)
    view = ChainlitTurnView(
        detailed_tools=bool(cl.user_session.get("show_tool_details", False)),
        tool_title_resolver=lambda name, input_text: auxiliary_labels.describe_tool(
            chat_id, name, input_text
        ),
    )
    active_views[chat_id] = view
    try:
        if is_revision:
            async with chainlit_layer.revision(message.id):
                await view.start()
                answer = await _run_turn(
                    view,
                    runtime.revise(chat_id, message.id, message.content, view.handle),
                )
                await _complete_turn(view, chat_id, before_files, answer)
        else:
            await view.start()
            answer = await _run_turn(
                view,
                runtime.submit(chat_id, message.id, message.content, view.handle),
            )
            await _complete_turn(view, chat_id, before_files, answer)
    except _TurnRunFailed:
        return
    finally:
        active_views.pop(chat_id, None)


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
    if chat_id not in deleting_chats:
        binding = chat_configurations.current(chat_id)
        if binding is not None:
            binding = await chat_configurations.recover(
                chat_id,
                binding.profile_id,
            )
            if binding.mode_locked:
                await _sync_chat_configuration(chat_id, binding, refresh=True)
            else:
                _remember_chat_configuration(binding)
    view = active_views.get(chat_id)
    if view is not None:
        try:
            await view.cancel()
        except RuntimeError:
            pass


@router.get("/files/{object_key:path}", include_in_schema=False)
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


# Chainlit registers its SPA fallback before loading the user module. Keep this
# specific route ahead of that catch-all so persisted element URLs return bytes.
_local_file_route = router.routes.pop()
router.routes.insert(0, _local_file_route)


@app.on_event("shutdown")
async def close_resources() -> None:
    pending_titles = list(chat_title_tasks.values())
    for task in pending_titles:
        task.cancel()
    if pending_titles:
        await asyncio.gather(*pending_titles, return_exceptions=True)
    await agent_execution.close()
