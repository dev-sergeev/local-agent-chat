import os
import subprocess
from pathlib import Path


def test_run_script_loads_and_exports_dotenv_itself(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CHAINLIT_AUTH_SECRET=a-test-secret-that-is-at-least-32-characters\n"
        "MODEL_PROFILES_FILE=models.yaml\n"
        "APP_DATA_DIR=.data\n"
        "APP_PORT=8765\n"
        'APP_ROOT_PATH="${JUPYTERHUB_SERVICE_PREFIX%/}/vscode/proxy/$APP_PORT"\n',
        encoding="utf-8",
    )
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    fake_chainlit = binary_dir / "chainlit"
    fake_chainlit.write_text(
        "#!/usr/bin/env bash\n"
        'test -n "$CHAINLIT_AUTH_SECRET"\n'
        'test "$APP_ROOT_PATH" = "/user/test/vscode/proxy/8765"\n'
        'printf "%s" "$*"\n',
        encoding="utf-8",
    )
    fake_chainlit.chmod(0o755)
    env = {
        "PATH": f"{binary_dir}:{os.environ['PATH']}",
        "ENV_FILE": str(env_file),
        "JUPYTERHUB_SERVICE_PREFIX": "/user/test/",
    }

    result = subprocess.run(
        ["bash", "scripts/run.sh"],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.replace(
        "a-test-secret-that-is-at-least-32-characters", "<REDACTED>"
    )
    assert "--port 8765 --root-path /user/test/vscode/proxy/8765" in result.stdout
    assert "--host 127.0.0.1" in result.stdout
    assert "--headless" in result.stdout
