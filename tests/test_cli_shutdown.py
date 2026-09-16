import os
import signal
import socket
import subprocess
import sys
import time

import pytest
import requests


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
def test_cli_signal_runs_final_cleanup(tmp_path, stop_signal):
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("APP_", "MODEL_", "LOCALCHAT_", "CHAINLIT_", "OPENAI_"))
    }
    environment.update(
        XDG_CONFIG_HOME=str(tmp_path / "config"), XDG_DATA_HOME=str(tmp_path / "data")
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "local_agent_chat",
            "init",
            "--no-input",
            "--model",
            "openai:test",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--no-api-key",
        ],
        env=environment,
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        # A recently stopped HTTP/WebSocket server may leave TIME_WAIT sockets.
        # The CLI must accept the same reusable port that Uvicorn accepts.
        probe.listen()
        with socket.socket() as client:
            client.connect(("127.0.0.1", port))
            peer, _ = probe.accept()
            peer.close()
            assert client.recv(1) == b""
    with (tmp_path / "server.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "local_agent_chat", "run", "--port", str(port)],
            env=environment,
            cwd=tmp_path,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(150):
                assert process.poll() is None, (tmp_path / "server.log").read_text()
                try:
                    if (
                        requests.get(
                            f"http://127.0.0.1:{port}/", timeout=0.2
                        ).status_code
                        == 200
                    ):
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.1)
            else:
                pytest.fail("CLI server did not start")
            process.send_signal(stop_signal)
            process.wait(timeout=15)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    data = tmp_path / ".local-agent-chat"
    assert list(data.glob("*.sqlite3")), "Persistent data should survive shutdown"
    assert list(data.glob(".runtime-*")) == [], (
        f"Temporary workspace survived signal {stop_signal}; process returned {process.returncode}"
    )
