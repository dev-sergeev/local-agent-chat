from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from hashlib import sha256
from pathlib import Path

import chainlit as cl
from chainlit.config import config as chainlit_config
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
    chat_title_source,
    fallback_chat_title,
)
from local_agent_chat.deep_agent_execution import DeepAgentExecution
from local_agent_chat.llm_retry import RetryBlock
from local_agent_chat.local_storage import LocalStorageClient
from local_agent_chat.long_term_memory import MarkdownMemory
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
public_dir = Path(__file__).resolve().parent / "public"
logo_path = public_dir / "localchat-logo.png"
avatar_path = public_dir / "avatars" / "localchat.png"
logo_version = sha256(logo_path.read_bytes()).hexdigest()
avatar_version = sha256(avatar_path.read_bytes()).hexdigest()
# Chainlit's theme logo endpoint is stable across asset changes. Publish the
# same file under a content-versioned URL so an ordinary reload invalidates it.
chainlit_config.ui.logo_file_url = (
    f"{settings.root_path}/public/{logo_path.name}?v={logo_version}"
)
# Publish the avatar through an explicit content-versioned URL as well, so
# browser caches cannot keep an outdated brand image after an asset update.
chainlit_config.ui.default_avatar_file_url = (
    f"{settings.root_path}/public/avatars/{avatar_path.name}?v={avatar_version}"
)
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
project_skills_dir = Path(__file__).resolve().parent / "skills"
sandbox_manager = LocalSandboxManager(
    sandbox_files,
    system_read_roots=(project_skills_dir,),
)
runtime_history = SQLiteHistory(settings.data_dir / "runtime-history.sqlite3")
long_term_memory = MarkdownMemory(settings.data_dir / "memory" / "MEMORY.md")
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
    long_term_memory=long_term_memory,
    retry_block=retry_block,
    skills_dir=project_skills_dir,
    chat_bindings=chat_bindings,
)
chat_configurations = ChatConfigurations(
    chat_bindings,
    (model.id for model in settings.models),
    chainlit_layer.has_user_request,
)
install_mode_acceptance_guard(
    lambda chat_id, profile, host_files: chat_configurations.select_mode(
        chat_id,
        AgentMode.HOST_FILES if host_files else AgentMode.CHAT_FILES,
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
            message=(
                "Изучи файлы, загруженные в этот диалог, и кратко опиши их "
                "структуру и назначение."
            ),
        ),
        cl.Starter(
            label="Найти причину",
            message=(
                "Найди причину проблемы в приложенных файлах и предложи точное "
                "исправление в ответе."
            ),
        ),
        cl.Starter(
            label="Проанализировать данные",
            message=(
                "Изучи приложенные файлы, проанализируй данные и сформулируй "
                "результаты в чате."
            ),
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
        fallback_title = fallback_chat_title(request_text)
        await cl.context.emitter.emit(
            "first_interaction",
            {"interaction": fallback_title, "thread_id": chat_id},
        )
        title = await auxiliary_labels.describe_chat(chat_id, request_text)
        rendered = title or fallback_title
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
                id="host_files_access",
                label="Чтение файлов с диска",
                initial=binding.mode is AgentMode.HOST_FILES,
                tooltip=(
                    "Разрешает чтение файлов с диска. Создание, изменение и "
                    "удаление файлов, а также выполнение команд недоступны в "
                    "обоих режимах. Выбор фиксируется после первого сообщения."
                ),
                description=(
                    "Выключено: только файлы, загруженные в этот диалог. "
                    "Включено: чтение и поиск по абсолютным путям, доступным "
                    "процессу приложения."
                ),
                disabled=binding.mode_locked,
            ),
            Switch(
                id="show_tool_details",
                label="Подробные результаты инструментов",
                initial=detailed,
                tooltip="Показывать более полные результаты чтения, поиска и памяти.",
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
        AgentMode.HOST_FILES
        if updated.get("host_files_access") is True
        else AgentMode.CHAT_FILES
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


async def _run_turn(view: ChainlitTurnView, operation: Awaitable[str]) -> str:
    try:
        return await operation
    except asyncio.CancelledError:
        await asyncio.shield(view.cancel())
        raise
    except Exception as error:  # noqa: BLE001 - provider/tool failures end the Turn
        await view.fail(safe_text(error, max_chars=2000))
        raise _TurnRunFailed from error


async def _handle_message(message: cl.Message, chat_id: str) -> None:
    binding = chat_configurations.accept_message(
        chat_id,
        cl.user_session.get("chat_profile"),
    )
    await _sync_chat_configuration(chat_id, binding, refresh=True)

    is_revision = await runtime.has_turn(message.id)
    is_first_turn = not is_revision and not await runtime_history.has_chat(chat_id)
    uploads: list[tuple[Path, str]] = []
    for element in message.elements or []:
        source = getattr(element, "path", None)
        if source:
            source_path = Path(source)
            uploads.append((source_path, element.name or source_path.name))

    title_state = await chainlit_layer.chat_title_state(chat_id)
    should_title = False
    if is_first_turn:
        should_title = await chainlit_layer.begin_chat_title(chat_id)
    elif title_state in {CHAT_TITLE_PENDING, CHAT_TITLE_FALLBACK}:
        should_title = await chainlit_layer.begin_chat_title(chat_id)
    if should_title:
        current_title_source = chat_title_source(
            message.content, [name for _source, name in uploads]
        )
        title_source = (
            current_title_source
            if is_first_turn
            else await chainlit_layer.first_user_request(chat_id)
            or current_title_source
        )
        _start_chat_title(chat_id, title_source)

    async def upload_message_files() -> None:
        for source_path, name in uploads:
            await sandbox_files.upload(chat_id, source_path, name)

    if not is_revision:
        await upload_message_files()

    view = ChainlitTurnView(
        detailed_tools=bool(cl.user_session.get("show_tool_details", False)),
    )
    active_views[chat_id] = view
    try:
        if is_revision:
            async with runtime.revision_transaction(
                chat_id,
                message.id,
                message.content,
                view.handle,
                before_run=upload_message_files if uploads else None,
            ) as run_revision:
                async with chainlit_layer.revision(message.id):
                    await view.start()
                    answer = await _run_turn(view, run_revision())
                    await view.complete(answer)
        else:
            await view.start()
            answer = await _run_turn(
                view,
                runtime.submit(chat_id, message.id, message.content, view.handle),
            )
            await view.complete(answer)
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


async def close_resources() -> None:
    pending_titles = list(chat_title_tasks.values())
    for task in pending_titles:
        task.cancel()
    if pending_titles:
        await asyncio.gather(*pending_titles, return_exceptions=True)
    await agent_execution.close()


if getattr(app.state, "_localchat_base_lifespan", None) is None:
    app.state._localchat_base_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def localchat_lifespan(application):
        try:
            async with application.state._localchat_base_lifespan(application) as state:
                yield state
        finally:
            cleanup = getattr(application.state, "_localchat_close_resources", None)
            if cleanup is not None:
                await cleanup()

    app.router.lifespan_context = localchat_lifespan

app.state._localchat_close_resources = close_resources
