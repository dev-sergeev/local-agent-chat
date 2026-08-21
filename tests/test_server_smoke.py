import os
import socket
import subprocess
import time
from pathlib import Path

import requests


def test_chainlit_server_works_behind_root_path(tmp_path: Path) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    prefix = f"/user/test/vscode/proxy/{port}"
    env = os.environ | {
        "APP_ROOT_PATH": prefix,
        "APP_DATA_DIR": str(tmp_path / "data"),
        "MODEL_PROFILES_FILE": str(Path("models.example.yaml").resolve()),
        "LOCAL_MODEL_API_KEY": "dummy",
        "OPENAI_API_KEY": "dummy",
        "CHAINLIT_AUTH_SECRET": "a-secure-smoke-test-secret-that-is-long-enough",
    }
    process = subprocess.Popen(
        [
            "chainlit",
            "run",
            "app.py",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--root-path",
            prefix,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    session = requests.Session()
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
        stripped_response = session.get(
            f"http://127.0.0.1:{port}/", timeout=2
        )
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
        settings = session.get(f"{root}/project/settings", timeout=2).json()
        assert settings["dataPersistence"] is True
        assert settings["threadResumable"] is True
        assert settings["features"]["edit_message"] is True
        assert settings["features"]["spontaneous_file_upload"]["max_size_mb"] == 100
        assert settings["ui"]["cot"] == "tool_call"
        assert settings["ui"]["language"] == "ru-RU"
        assert settings["ui"]["layout"] == "wide"
        assert [item["name"] for item in settings["chatProfiles"]] == [
            "openrouter-deepseek"
        ]
        assert len(settings["chatProfiles"][0]["starters"]) == 4
        translations = session.get(f"{root}/project/translations", timeout=2).json()
        assert translations["translation"]["chat"]["input"]["placeholder"] == (
            "Напишите сообщение..."
        )
    finally:
        process.terminate()
        process.wait(timeout=10)
