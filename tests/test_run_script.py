import os
import subprocess
from pathlib import Path


def test_run_script_delegates_literal_dotenv_to_cli(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    marker = tmp_path / "shell-was-executed"
    env_file.write_text(f'TOKEN="$(touch {marker})"\n', encoding="utf-8")
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    fake_python = binary_dir / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\ntest -z "${TOKEN:-}"\nprintf "%s\\n" "$@"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = {"PATH": f"{binary_dir}:{os.environ['PATH']}", "ENV_FILE": str(env_file)}
    result = subprocess.run(
        ["bash", str(Path(__file__).parents[1] / "scripts/run.sh"), "--port", "9000"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "-m",
        "local_agent_chat",
        "run",
        "--config-dir",
        str(tmp_path),
        "--port",
        "9000",
    ]
    assert not marker.exists()
