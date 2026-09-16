"""Install a wheel in a clean venv; exercise its UI protocol and persisted chat.

Run with the test dependencies installed in the invoking interpreter. The fresh
application environment receives only the wheel's normal runtime dependencies.
All provider requests go to a deterministic local OpenAI/GigaChat test server.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
import venv
import zipfile
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests
import socketio


class Provider(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = request["messages"]
        latest = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        message = {"role": "assistant", "content": "Answer: " + str(latest)}
        finish = "stop"
        if (
            request.get("tools") or request.get("functions")
        ) and latest == "read note.txt":
            if messages[-1]["role"] in {"tool", "function"}:
                message["content"] = "File contents: " + messages[-1]["content"]
            else:
                finish = "tool_calls"
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_read_note",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"file_path":"/note.txt"}',
                            },
                        }
                    ],
                }
                if request.get("functions"):
                    finish = "function_call"
                    message = {
                        "role": "assistant",
                        "content": "",
                        "function_call": {
                            "name": "read_file",
                            "arguments": {"file_path": "/note.txt"},
                        },
                    }
        body = json.dumps(
            {
                "id": "chatcmpl-local-test",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "test-model",
                "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def wait_for(condition, description, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = condition()
            if result:
                return result
        except (requests.RequestException, sqlite3.OperationalError):
            pass
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for {description}")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def verify_history(data: Path, thread: str, expected: list[tuple[str, str]]):
    with closing(sqlite3.connect(data / "runtime-history.sqlite3")) as db:
        turns = db.execute(
            "SELECT id, text, answer FROM turns WHERE chat_id=? ORDER BY sequence",
            (thread,),
        ).fetchall()
    assert [(row[0], row[1]) for row in turns] == expected, turns
    assert all(row[2] for row in turns), turns
    with closing(sqlite3.connect(data / "chainlit.sqlite3")) as db:
        steps = db.execute(
            'SELECT id, output, type FROM steps WHERE "threadId"=? ORDER BY "stepOrder"',
            (thread,),
        ).fetchall()
        assert [(r[0], r[1]) for r in steps if r[2] == "user_message"] == expected
        assert [r[1] for r in steps if r[2] == "assistant_message"] == [
            r[2] for r in turns
        ]
        for table in ("step_revisions", "element_revisions", "feedback_revisions"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    for path in data.glob("*.sqlite3"):
        with closing(sqlite3.connect(path)) as db:
            assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    return turns


class Chat:
    def __init__(self, root, prefix, thread=None):
        self.root = root
        self.session_id = str(uuid.uuid4())
        self.thread = thread
        self.resumed = None
        self.ended = threading.Event()
        self.ready = threading.Event()
        self.http = requests.Session()
        self.http.post(root + "/auth/header", timeout=10).raise_for_status()
        self.ws = socketio.Client(http_session=self.http, reconnection=False)
        self.ws.on("task_end", lambda *_: self.ended.set())
        self.ws.on("chat_settings", lambda *_: self.ready.set())
        self.ws.on("first_interaction", self._thread)
        self.ws.on("resume_thread", self._resume)
        self.ws.connect(
            root.removesuffix(prefix),
            socketio_path=prefix + "/ws/socket.io",
            transports=["websocket"],
            auth={
                "sessionId": self.session_id,
                "userEnv": "{}",
                "clientType": "webapp",
                "threadId": thread,
            },
            wait_timeout=10,
        )
        self.ws.emit("connection_successful")
        assert self.ready.wait(15), "Chat settings did not arrive"
        if thread:
            wait_for(lambda: self.resumed, "restored UI timeline")

    def _thread(self, event):
        self.thread = event["thread_id"]

    def _resume(self, event):
        self.resumed = event

    def send(self, text, *, edit_id=None, file=None):
        identifier = edit_id or str(uuid.uuid4())
        payload = {
            "message": {
                "id": identifier,
                "output": text,
                "type": "user_message",
                "name": "User",
                "createdAt": datetime.now(timezone.utc).isoformat(),
            }
        }
        if file:
            upload = self.http.post(
                self.root + "/project/file",
                params={"session_id": self.session_id},
                files={"file": ("note.txt", b"PACKAGE-CANARY", "text/plain")},
                timeout=10,
            )
            upload.raise_for_status()
            payload["fileReferences"] = [{"id": upload.json()["id"]}]
        self.ended.clear()
        self.ws.emit("edit_message" if edit_id else "client_message", payload)
        assert self.ended.wait(45), "Chat turn did not finish"
        return identifier, text

    def close(self):
        self.ws.disconnect()
        self.http.close()


@contextmanager
def service(executable, config, port, prefix, outside, env, log_path):
    config_arguments = [] if config == outside else ["--config-dir", str(config)]
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                str(executable),
                "run",
                *config_arguments,
                "--port",
                str(port),
                "--root-path",
                prefix,
            ],
            cwd=outside,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        root = f"http://127.0.0.1:{port}{prefix}"
        try:

            def available():
                assert process.poll() is None, log_path.read_text()
                response = requests.get(root + "/", timeout=1)
                return response if response.status_code == 200 else None

            response = wait_for(available, "installed application startup", timeout=60)
            yield root, response.text
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                raise AssertionError("Application did not shut down gracefully")


def check(wheel: Path, work: Path):
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for suffix in (
            "app.py",
            "cli.py",
            "assets/chainlit/config.toml",
            "assets/chainlit/translations/ru-RU.json",
            "assets/public/branding.css",
            "assets/public/proxy-method-override.js",
            "assets/chainlit.md",
            "assets/examples/models.example.yaml",
        ):
            assert "local_agent_chat/" + suffix in names, suffix
        assert not any(
            name.endswith((".sqlite3", "/.env", "/models.yaml", ".log"))
            for name in names
        )
    application_env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(
            (
                "APP_",
                "CHAINLIT_",
                "LOCALCHAT_",
                "MODEL_",
                "OPENAI_",
                "GIGACHAT_",
                "JUPYTERHUB_",
                "AGENT_",
                "LLM_",
                "PYTHONPATH",
            )
        )
    }
    application_env.update(
        XDG_CONFIG_HOME=str(work / "config"), XDG_DATA_HOME=str(work / "data")
    )
    outside = work / "outside"
    outside.mkdir(parents=True)
    environment = work / "venv"
    print("Creating isolated environment and installing the wheel...", flush=True)
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin/python"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            str(wheel),
        ],
        cwd=outside,
        env=application_env,
        check=True,
        stdout=(work / "install.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        [str(python), "-m", "pip", "check"],
        cwd=outside,
        env=application_env,
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import local_agent_chat; from pathlib import Path; assert Path(local_agent_chat.__file__).is_relative_to(Path(__import__('sys').prefix)); print(local_agent_chat.__file__)",
        ],
        cwd=outside,
        env=application_env,
        check=True,
    )
    executable = environment / "bin/localchat"
    subprocess.run(
        [str(executable), "--version"], cwd=outside, env=application_env, check=True
    )
    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    threading.Thread(target=provider.serve_forever, daemon=True).start()
    config = outside
    data = outside / ".local-agent-chat"
    try:
        subprocess.run(
            [
                str(executable),
                "init",
                "--no-input",
                "--model",
                "openai:test-model",
                "--base-url",
                f"http://127.0.0.1:{provider.server_port}/v1",
                "--no-api-key",
                "--no-streaming",
            ],
            cwd=outside,
            env=application_env,
            check=True,
        )
        original_config = (config / ".env").read_bytes()
        port = free_port()
        prefix = f"/user/test/vscode/proxy/{port}"
        with service(
            executable,
            config,
            port,
            prefix,
            outside,
            application_env,
            work / "server-first.log",
        ) as (root, html):
            paths = re.findall(r'(?:src|href)="([^\"]+)"', html)
            assets = [
                path for path in paths if "/assets/" in path or "/public/" in path
            ]
            assert any(path.endswith(".js") for path in assets)
            for path in assets:
                response = requests.get(root.removesuffix(prefix) + path, timeout=10)
                assert response.status_code == 200, path
                assert not response.headers.get("content-type", "").startswith(
                    "text/html"
                ), path
            chat = Chat(root, prefix)
            try:
                expected = [chat.send("read note.txt", file=True)]
                for index in range(2, 6):
                    expected.append(chat.send(f"request-{index}"))
                turns = verify_history(data, chat.thread, expected)
                assert "PACKAGE-CANARY" in turns[0][2]
                expected = expected[:3]
                expected[2] = chat.send("revised-third", edit_id=expected[2][0])
                verify_history(data, chat.thread, expected)
                thread = chat.thread
            finally:
                chat.close()
        assert not list(data.glob(".runtime-*")), "Temporary runtime survived shutdown"
        print(
            "Wheel UI assets, upload/tool use and historical edit passed; reinstalling...",
            flush=True,
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                str(wheel),
            ],
            cwd=outside,
            env=application_env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        assert (config / ".env").read_bytes() == original_config
        # Relative configuration must survive moving a complete Chat directory.
        moved = work / "moved-project"
        outside.rename(moved)
        outside.mkdir()
        config = moved
        data = moved / ".local-agent-chat"
        with service(
            executable,
            config,
            port,
            "",
            outside,
            application_env,
            work / "server-restarted.log",
        ) as (root, _html):
            chat = Chat(root, "", thread)
            try:
                assert [
                    (s["id"], s["output"])
                    for s in chat.resumed["steps"]
                    if s["type"] == "user_message"
                ] == expected
                verify_history(data, thread, expected)
                expected.append(chat.send("after-reinstall"))
                expected[0] = chat.send("revised-first", edit_id=expected[0][0])
                expected = expected[:1]
                verify_history(data, thread, expected)
                with closing(sqlite3.connect(data / "checkpoints.sqlite3")) as db:
                    context = db.execute(
                        "SELECT messages FROM agent_context WHERE chat_id=?", (thread,)
                    ).fetchone()[0]
                    assert (
                        "revised-third" not in context
                        and "after-reinstall" not in context
                    )
            finally:
                chat.close()
        assert {path.name for path in config.iterdir()} == {
            ".env",
            "models.yaml",
            ".local-agent-chat",
        }
        assert not list(outside.iterdir())
        assert not (work / "config").exists()
        assert not (work / "data").exists()
        assert not list(data.glob(".runtime-*"))
        giga_project = work / "gigachat-project"
        giga_project.mkdir()
        giga_env = application_env | {"GIGACHAT_ACCESS_TOKEN": "test-access-token"}
        subprocess.run(
            [
                str(executable),
                "init",
                "--no-input",
                "--model",
                "gigachat:GigaChat-2",
                "--base-url",
                f"http://127.0.0.1:{provider.server_port}/v1",
                "--no-streaming",
            ],
            cwd=giga_project,
            env=giga_env,
            check=True,
        )
        # Verify that run reads the saved token instead of relying on the parent.
        with service(
            executable,
            giga_project,
            free_port(),
            "",
            giga_project,
            application_env,
            work / "server-gigachat.log",
        ) as (root, _html):
            chat = Chat(root, "")
            try:
                expected = [chat.send("read note.txt", file=True)]
                turns = verify_history(
                    giga_project / ".local-agent-chat", chat.thread, expected
                )
                assert "PACKAGE-CANARY" in turns[0][2]
            finally:
                chat.close()
        for log in work.glob("server-*.log"):
            text = log.read_text()
            assert not any(
                marker in text
                for marker in (
                    "Traceback",
                    "OperationalError",
                    "IntegrityError",
                    "Task exception was never retrieved",
                )
            ), log
        (work / "result.json").write_text(
            json.dumps(
                {
                    "wheel": wheel.name,
                    "passed": True,
                    "checks": [
                        "isolated installation",
                        "UI resources through proxy",
                        "upload and sandbox tool",
                        "five-turn history",
                        "edit third request",
                        "reinstallation",
                        "moved project directory",
                        "native GigaChat tools",
                        "resume",
                        "edit first request",
                        "SQLite integrity",
                        "current context",
                        "shutdown",
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"Installed distribution checks passed. Logs and evidence: {work}",
            flush=True,
        )
    finally:
        provider.shutdown()
        provider.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--work-dir", type=Path)
    arguments = parser.parse_args()
    directory = arguments.work_dir or Path(tempfile.mkdtemp(prefix="localchat-wheel-"))
    directory.mkdir(parents=True, exist_ok=True)
    check(arguments.wheel.resolve(), directory.resolve())
