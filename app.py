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
from local_agent_chat.agent_service import AgentService
from local_agent_chat.chainlit_data import create_chainlit_data_layer
from local_agent_chat.chainlit_ui import ChainlitTurnView
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
agent_service = AgentService(
    settings.data_dir / "checkpoints.sqlite3", settings.models, sandbox_manager
)
runtime_history = SQLiteHistory(settings.data_dir / "runtime-history.sqlite3")
runtime = ChatRuntime(
    agent=agent_service,
    sandbox=sandbox_files,
    history=runtime_history,
)
active_views: dict[str, ChainlitTurnView] = {}


async def cleanup_chat(chat_id: str) -> None:
    await agent_service.delete_chat(chat_id)
    await sandbox_manager.delete_chat(chat_id)
    await sandbox_files.delete_chat(chat_id)
    await runtime_history.delete_chat(chat_id)


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


async def _send_chat_settings() -> None:
    detailed = bool(cl.user_session.get("show_tool_details", False))
    await cl.ChatSettings(
        [
            Switch(
                id="show_tool_details",
                label="Подробные результаты инструментов",
                initial=detailed,
                tooltip="Показывать больше stdout, stderr и результатов файловых операций.",
            )
        ]
    ).send()


@cl.on_chat_start
async def on_chat_start():
    chat_id = _thread_id()
    profile_id = _selected_profile(chat_id)
    agent_service.set_profile(chat_id, profile_id)
    cl.user_session.set("model_profile", profile_id)
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
    cl.user_session.set("model_profile", profile_id)
    await _send_chat_settings()


@cl.on_settings_update
async def on_settings_update(updated):
    cl.user_session.set(
        "show_tool_details", bool(updated.get("show_tool_details", False))
    )


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


@cl.on_message
async def on_message(message: cl.Message):
    chat_id = _thread_id()
    profile_id = _selected_profile(chat_id)
    agent_service.set_profile(chat_id, profile_id)
    await chainlit_layer.update_thread(chat_id, metadata={"model_profile": profile_id})

    for element in message.elements or []:
        source = getattr(element, "path", None)
        if source:
            await sandbox_files.upload(
                chat_id, Path(source), element.name or Path(source).name
            )

    before_files = sandbox_files.manifest(chat_id)
    is_revision = await runtime.has_turn(message.id)
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


@cl.on_stop
async def on_stop():
    view = active_views.get(_thread_id())
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
    await agent_service.close()
