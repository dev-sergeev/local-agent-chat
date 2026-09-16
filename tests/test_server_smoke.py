import os
import socket
import sqlite3
import subprocess
import sys
import time
import uuid
from hashlib import sha256
from pathlib import Path

import pytest
import requests
import socketio

from local_agent_chat.cli import PROXY_TEMPLATE, reserve_port


@pytest.fixture(params=["static", "auto", PROXY_TEMPLATE])
def server_address(request):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        start = busy.getsockname()[1]
        if request.param == "static":
            busy.close()
            yield start, start, "static"
        else:
            busy.listen()
            with reserve_port("127.0.0.1", start) as candidate:
                port = candidate.getsockname()[1]
            yield start, port, request.param


def test_chainlit_server_works_behind_root_path(tmp_path: Path, server_address) -> None:
    start, port, mode = server_address
    prefix = f"/user/test/vscode/proxy/{port}"
    env = os.environ | {
        "APP_ROOT_PATH": prefix if mode == "static" else mode,
        "JUPYTERHUB_SERVICE_PREFIX": "/user/test/",
        "APP_DATA_DIR": str(tmp_path / "data"),
        "MODEL_PROFILES_FILE": str(Path("models.example.yaml").resolve()),
        "LOCAL_MODEL_API_KEY": "dummy",
        "OPENAI_API_KEY": "dummy",
        "CHAINLIT_AUTH_SECRET": "a-secure-smoke-test-secret-that-is-long-enough",
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "local_agent_chat",
            "run",
            "--config-dir",
            str(tmp_path / "config"),
            "--host",
            "127.0.0.1",
            "--port",
            str(start),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    session = requests.Session()
    websocket = socketio.Client(
        http_session=session,
        reconnection=False,
        logger=False,
        engineio_logger=False,
    )
    websocket_session_id = str(uuid.uuid4())
    try:
        root = f"http://127.0.0.1:{port}{prefix}"
        for _ in range(120):
            try:
                response = session.get(f"{root}/", timeout=0.5)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.1)
        else:
            raise AssertionError("server did not start within 12 seconds")
        assert prefix in response.text
        assert f"{prefix}/public/proxy-method-override.js" in response.text
        assert f"{prefix}/public/branding.css" in response.text
        proxy_script = session.get(f"{root}/public/proxy-method-override.js", timeout=2)
        assert proxy_script.status_code == 200
        assert 'headers.set("X-Proxy-Method-Override", "DELETE")' in (proxy_script.text)
        stripped_response = session.get(f"http://127.0.0.1:{port}/", timeout=2)
        assert stripped_response.status_code == 200
        assert prefix in stripped_response.text
        socket_response = session.get(
            f"http://127.0.0.1:{port}/ws/socket.io/",
            params={"EIO": "4", "transport": "polling"},
            timeout=2,
        )
        assert socket_response.status_code == 200
        assert socket_response.text.startswith("0{")
        assert session.post(f"{root}/auth/header", timeout=2).status_code == 200
        persisted_blob = tmp_path / "data" / "blobs" / "smoke" / "persisted.txt"
        persisted_blob.parent.mkdir(parents=True, exist_ok=True)
        persisted_blob.write_bytes(b"persisted attachment")
        persisted_download = session.get(f"{root}/files/smoke/persisted.txt", timeout=2)
        assert persisted_download.status_code == 200
        assert persisted_download.content == b"persisted attachment"
        assert persisted_download.headers["content-type"].startswith("text/plain")
        websocket.connect(
            f"http://127.0.0.1:{port}",
            socketio_path="/ws/socket.io",
            transports=["websocket"],
            auth={
                "sessionId": websocket_session_id,
                "userEnv": "{}",
                "clientType": "webapp",
            },
            wait_timeout=2,
        )
        upload = session.post(
            f"{root}/project/file",
            params={"session_id": websocket_session_id},
            files={"file": ("script.py", b"", "text/x-python")},
            timeout=2,
        )
        assert upload.status_code == 200
        file_id = upload.json()["id"]
        download = session.get(
            f"{root}/project/file/{file_id}",
            params={"session_id": websocket_session_id},
            timeout=2,
        )
        assert download.status_code == 200
        assert download.content == b""
        database = tmp_path / "data" / "chainlit.sqlite3"
        with sqlite3.connect(database) as connection:
            user_id = connection.execute(
                "SELECT id FROM users WHERE identifier = 'local-user'"
            ).fetchone()[0]
            connection.execute(
                """INSERT INTO threads
                   (id, "createdAt", name, "userId", "userIdentifier", metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    "delete-through-proxy",
                    "2026-08-25T00:00:00Z",
                    "Delete through proxy",
                    user_id,
                    "local-user",
                    "{}",
                ),
            )
        delete_response = session.post(
            f"{root}/project/thread",
            headers={"X-Proxy-Method-Override": "DELETE"},
            json={"threadId": "delete-through-proxy"},
            timeout=2,
        )
        assert delete_response.status_code == 200
        assert delete_response.json() == {"success": True}
        with sqlite3.connect(database) as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM threads WHERE id = ?",
                    ("delete-through-proxy",),
                ).fetchone()[0]
                == 0
            )
        settings = session.get(f"{root}/project/settings", timeout=2).json()
        assert settings["dataPersistence"] is True
        assert settings["threadResumable"] is True
        assert settings["features"]["edit_message"] is True
        assert settings["features"]["spontaneous_file_upload"]["max_size_mb"] == 100
        assert settings["features"]["spontaneous_file_upload"]["accept"] == []
        assert settings["ui"]["cot"] == "tool_call"
        assert settings["ui"]["language"] == "ru-RU"
        assert settings["ui"]["layout"] == "wide"
        assert settings["ui"]["name"] == "LocalChat"
        assert settings["ui"]["custom_css"] == "/public/branding.css"
        branding = session.get(f"{root}/public/branding.css", timeout=2)
        assert branding.status_code == 200
        assert "--localchat-brand-primary: #E13662" in branding.text
        assert "var(--localchat-brand-primary)" in branding.text
        assert "width: min(420px, calc(100vw - 3rem))" in branding.text
        wordmark_bytes = Path(
            "local_agent_chat/assets/public/localchat-logo.png"
        ).read_bytes()
        avatar_bytes = Path(
            "local_agent_chat/assets/public/avatars/localchat.png"
        ).read_bytes()
        favicon_bytes = Path("local_agent_chat/assets/public/favicon.png").read_bytes()
        logo_version = sha256(wordmark_bytes).hexdigest()
        avatar_version = sha256(avatar_bytes).hexdigest()
        versioned_logo_url = f"{prefix}/public/localchat-logo.png?v={logo_version}"
        versioned_avatar_url = (
            f"{prefix}/public/avatars/localchat.png?v={avatar_version}"
        )
        assert settings["ui"]["logo_file_url"] == versioned_logo_url
        assert settings["ui"]["default_avatar_file_url"] == versioned_avatar_url
        versioned_logo = session.get(
            f"http://127.0.0.1:{port}{versioned_logo_url}", timeout=2
        )
        assert versioned_logo.status_code == 200
        assert versioned_logo.content == wordmark_bytes
        logo = session.get(f"{root}/logo?theme=dark", timeout=2)
        assert logo.status_code == 200
        assert logo.content == wordmark_bytes
        light_logo = session.get(f"{root}/logo?theme=light", timeout=2)
        assert light_logo.status_code == 200
        assert light_logo.content == wordmark_bytes
        avatar = session.get(
            f"http://127.0.0.1:{port}{versioned_avatar_url}", timeout=2
        )
        assert avatar.status_code == 200
        assert avatar.content == avatar_bytes
        favicon = session.get(f"{root}/favicon", timeout=2)
        assert favicon.status_code == 200
        assert favicon.content == favicon_bytes
        assert [item["name"] for item in settings["chatProfiles"]] == [
            "openrouter-deepseek"
        ]
        assert len(settings["chatProfiles"][0]["starters"]) == 4
        translations = session.get(f"{root}/project/translations", timeout=2).json()
        assert translations["translation"]["chat"]["input"]["placeholder"] == (
            "Напишите сообщение..."
        )
    finally:
        if websocket.connected:
            websocket.emit("clear_session")
            websocket.disconnect()
        process.terminate()
        process.wait(timeout=10)
