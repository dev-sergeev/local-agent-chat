import errno
import os
import stat
import subprocess
import sys

import pytest
import yaml
from dotenv import dotenv_values

from local_agent_chat.cli import main
from local_agent_chat.installation import config_directory, runtime_workspace


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith(
            (
                "APP_",
                "CHAINLIT_",
                "LOCALCHAT_",
                "MODEL_",
                "OPENAI_",
                "GIGACHAT_",
                "AGENT_",
                "LLM_",
                "JUPYTERHUB_",
            )
        ):
            monkeypatch.delenv(name)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def init_arguments():
    return [
        "init",
        "--no-input",
        "--model",
        "openai:test-model",
        "--base-url",
        "http://localhost:9999/v1",
    ]


def test_init_preserves_literal_secrets_and_existing_configuration(
    isolated_env, monkeypatch, capsys
):
    secret = "private-key-with-'quotes'-and-${HOME}-and-\\slash"
    directory_mode = stat.S_IMODE(isolated_env.stat().st_mode)
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    assert main(init_arguments()) == 0
    directory = config_directory()
    env_file = directory / ".env"
    before = env_file.read_bytes()
    settings = dotenv_values(env_file, interpolate=False)
    assert settings["OPENAI_API_KEY"] == secret
    assert len(settings["CHAINLIT_AUTH_SECRET"]) >= 32
    assert directory == isolated_env
    assert settings["APP_DATA_DIR"] == ".local-agent-chat"
    assert settings["APP_GENERATE_CHAT_TITLES"] == "false"
    assert settings["APP_ROOT_PATH"] == "auto"
    assert (directory / ".local-agent-chat").is_dir()
    assert not (isolated_env / "config").exists()
    assert not (isolated_env / "data").exists()
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(directory.stat().st_mode) == directory_mode
    assert secret not in (directory / "models.yaml").read_text()
    assert main(init_arguments()) == 2
    assert env_file.read_bytes() == before
    assert secret not in capsys.readouterr().out


def test_interactive_init_generates_model_and_secret(isolated_env, monkeypatch):
    replies = iter(["openai:my-local-model", "http://localhost:9000/v1"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(replies))
    monkeypatch.setattr("getpass.getpass", lambda prompt: "a-private-api-key")
    assert main(["init"]) == 0
    profile = yaml.safe_load((config_directory() / "models.yaml").read_text())[
        "models"
    ][0]
    assert profile["model"] == "openai:my-local-model"
    assert profile["base_url"] == "http://localhost:9000/v1"
    assert "summary_options" not in profile
    assert profile["streaming"] is False
    assert profile["max_tokens"] == 128000


def test_noninteractive_init_requires_key_or_explicit_local_mode(isolated_env, capsys):
    assert main(init_arguments()) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    assert not (config_directory() / ".env").exists()
    assert not (config_directory() / "models.yaml").exists()
    assert main(init_arguments() + ["--no-api-key", "--no-streaming"]) == 0
    profile = yaml.safe_load((config_directory() / "models.yaml").read_text())[
        "models"
    ][0]
    assert profile["streaming"] is False


def test_init_can_override_output_limit_and_enable_streaming(isolated_env):
    assert (
        main(init_arguments() + ["--no-api-key", "--streaming", "--max-tokens", "4096"])
        == 0
    )
    profile = yaml.safe_load((isolated_env / "models.yaml").read_text())["models"][0]
    assert profile["streaming"] is True
    assert profile["max_tokens"] == 4096


@pytest.mark.parametrize("limit", ["0", "-1", "1000001"])
def test_invalid_output_limit_leaves_no_configuration(isolated_env, limit):
    assert main(init_arguments() + ["--no-api-key", "--max-tokens", limit]) == 2
    assert list(isolated_env.iterdir()) == []


def test_run_reports_missing_config_without_creating_chainlit_files(
    isolated_env, capsys
):
    assert main(["run"]) == 2
    assert "localchat init" in capsys.readouterr().err
    assert not (isolated_env / ".chainlit").exists()
    assert not (isolated_env / ".files").exists()


@pytest.mark.parametrize("data_setting", [None, "", ".localchat", "relative-data"])
@pytest.mark.parametrize("explicit_config", [False, True])
def test_run_resolves_paths_from_config_and_environment_overrides(
    isolated_env, monkeypatch, data_setting, explicit_config
):
    from argparse import Namespace

    from local_agent_chat import cli
    from local_agent_chat.settings import load_settings

    assert main(init_arguments() + ["--no-api-key"]) == 0
    directory = config_directory()
    if data_setting is not None:
        with (directory / ".env").open("a") as output:
            output.write(f"APP_DATA_DIR='{data_setting}'\n")
    if explicit_config:
        elsewhere = isolated_env / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("APP_PORT", "9876")
    monkeypatch.setenv("OPENAI_API_KEY", "override-key")
    captured = {}

    class CheckedStop(Exception):
        pass

    def check_settings():
        captured.update(os.environ)
        raise CheckedStop

    monkeypatch.setattr("local_agent_chat.settings.load_settings", check_settings)
    with pytest.raises(CheckedStop):
        cli.run(
            Namespace(
                config_dir=str(directory) if explicit_config else None,
                data_dir=None,
                host=None,
                port=None,
                root_path="/proxy/9876",
                open_browser=False,
            )
        )
    assert captured["MODEL_PROFILES_FILE"] == str(directory / "models.yaml")
    assert captured["APP_DATA_DIR"] == str(
        directory / (data_setting or ".local-agent-chat")
    )
    # The child reads these values again after CLI initialization.
    assert load_settings().data_dir == directory / (data_setting or ".local-agent-chat")
    assert captured["APP_PORT"] == "9876"
    assert captured["OPENAI_API_KEY"] == "override-key"
    assert captured["APP_ROOT_PATH"] == "/proxy/9876"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--port", "0"],
        ["--port", "65536"],
        ["--root-path", "${JUPYTERHUB_SERVICE_PREFIX}/proxy"],
    ],
)
def test_invalid_startup_settings_have_actionable_errors(
    isolated_env, arguments, capsys
):
    assert main(init_arguments() + ["--no-api-key"]) == 0
    assert main(["run", *arguments]) == 2
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("lock_error", [None, errno.ENOLCK, errno.EOPNOTSUPP])
def test_runtime_refreshes_assets_keeps_data_and_excludes_concurrent_process(
    tmp_path, monkeypatch, lock_error
):
    if lock_error is not None:

        def unavailable(*_args):
            raise OSError(lock_error, os.strerror(lock_error))

        monkeypatch.setattr("fcntl.flock", unavailable)
    data = tmp_path / "data"
    data.mkdir()
    history = data / "existing.sqlite3"
    history.write_bytes(b"existing data")
    roots = []
    for _ in range(2):
        with runtime_workspace(data) as root:
            roots.append(root)
            assert (root / ".chainlit/config.toml").is_file()
            assert (root / "public/branding.css").is_file()
            assert (root / "chainlit_ru-RU.md").is_file()
            with pytest.raises(ValueError, match="Another LocalChat"):
                with runtime_workspace(data):
                    pass
        assert not root.exists()
    assert roots[0] != roots[1]
    assert history.read_bytes() == b"existing data"


def test_runtime_lock_excludes_processes_with_different_flock_support(
    tmp_path, monkeypatch
):
    import fcntl

    flock = fcntl.flock

    def unavailable(*_args):
        raise OSError(errno.ENOLCK, os.strerror(errno.ENOLCK))

    monkeypatch.setattr(fcntl, "flock", unavailable)
    with runtime_workspace(tmp_path):
        monkeypatch.setattr(fcntl, "flock", flock)
        with pytest.raises(ValueError, match="Another LocalChat"):
            with runtime_workspace(tmp_path):
                pass
    with runtime_workspace(tmp_path):
        monkeypatch.setattr(fcntl, "flock", unavailable)
        with pytest.raises(ValueError, match="Another LocalChat"):
            with runtime_workspace(tmp_path):
                pass


@pytest.mark.parametrize("during_setup", [False, True])
def test_runtime_releases_directory_lock_after_failure(
    tmp_path, monkeypatch, during_setup
):
    def fail_copy(_destination):
        raise RuntimeError("copy failed")

    with monkeypatch.context() as patch:
        if during_setup:
            patch.setattr("local_agent_chat.installation.copy_ui", fail_copy)
        with pytest.raises(RuntimeError):
            with runtime_workspace(tmp_path):
                raise RuntimeError("runtime failed")
    assert not (tmp_path / ".localchat.lock.d").exists()
    assert not list(tmp_path.glob(".runtime-*"))
    with runtime_workspace(tmp_path):
        pass


def test_runtime_reports_stale_directory_lock_without_removing_it(tmp_path):
    guard = tmp_path / ".localchat.lock.d"
    guard.mkdir()
    with pytest.raises(ValueError, match="If all instances are stopped"):
        with runtime_workspace(tmp_path):
            pass
    assert guard.is_dir()


def test_runtime_does_not_ignore_unrelated_lock_errors(tmp_path, monkeypatch):
    def denied(*_args):
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr("fcntl.flock", denied)
    with pytest.raises(PermissionError):
        with runtime_workspace(tmp_path):
            pass
    assert not (tmp_path / ".localchat.lock.d").exists()


def test_help_and_version_do_not_initialize_chainlit(tmp_path):
    for arguments in (["--help"], ["--version"], ["init", "--help"], ["run", "--help"]):
        result = subprocess.run(
            [sys.executable, "-m", "local_agent_chat", *arguments],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "localchat" in result.stdout
        if "--help" in arguments:
            assert "GigaChat" in result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "model,key_name",
    [
        ("openai:deepseek/deepseek-v4-flash-0731", "OPENAI_API_KEY"),
        ("gigachat:GigaChat-2", "GIGACHAT_ACCESS_TOKEN"),
    ],
)
def test_init_provider_and_local_directory(isolated_env, monkeypatch, model, key_name):
    monkeypatch.setenv(key_name, "test-token")
    assert (
        main(
            [
                "init",
                "--no-input",
                "--model",
                model,
                "--base-url",
                "https://models.example/v1",
            ]
        )
        == 0
    )
    profile = yaml.safe_load((isolated_env / "models.yaml").read_text())["models"][0]
    env = dotenv_values(isolated_env / ".env", interpolate=False)
    assert profile["model"] == model
    assert profile["api_key_env"] == key_name
    assert profile["base_url"] == "https://models.example/v1"
    assert profile["streaming"] is False
    assert profile["max_tokens"] == 128000
    assert env[key_name] == "test-token"
    assert "test-token" not in (isolated_env / "models.yaml").read_text()


def test_init_gigachat_custom_token_variable(isolated_env, monkeypatch):
    monkeypatch.setenv("COMPANY_TOKEN", "custom-token")
    assert (
        main(
            [
                "init",
                "--no-input",
                "--model",
                "gigachat:GigaChat-2",
                "--base-url",
                "https://models.example/v1",
                "--api-key-env",
                "COMPANY_TOKEN",
            ]
        )
        == 0
    )
    assert dotenv_values(isolated_env / ".env")["COMPANY_TOKEN"] == "custom-token"


@pytest.mark.parametrize(
    "model,no_key",
    [
        ("other:model", True),
        ("openai:", True),
        ("gigachat:", True),
        ("gigachat:GigaChat-2", True),
        ("gigachat:GigaChat-2", False),
    ],
)
def test_init_invalid_provider_or_missing_gigachat_token_leaves_no_files(
    isolated_env, model, no_key
):
    arguments = [
        "init",
        "--no-input",
        "--model",
        model,
        "--base-url",
        "https://models.example/v1",
    ]
    if no_key:
        arguments.append("--no-api-key")
    assert main(arguments) == 2
    assert list(isolated_env.iterdir()) == []


def test_init_custom_directory_and_data_are_relative_to_config(isolated_env):
    assert (
        main(
            init_arguments()
            + ["--no-api-key", "--config-dir", "project", "--data-dir", "state"]
        )
        == 0
    )
    directory = isolated_env / "project"
    assert (directory / "state").is_dir()
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((directory / "models.yaml").stat().st_mode) == 0o600
    assert dotenv_values(directory / ".env")["APP_DATA_DIR"] == "state"


def test_two_projects_and_moved_configuration_are_independent(
    isolated_env, monkeypatch
):
    from local_agent_chat.settings import load_settings

    for name in ("first", "second"):
        assert main(init_arguments() + ["--no-api-key", "--config-dir", name]) == 0
    first = isolated_env / "first"
    second = isolated_env / "second"
    moved = isolated_env / "moved"
    (first / ".local-agent-chat" / "existing.sqlite3").write_bytes(b"persisted")
    first.rename(moved)
    assert (
        dotenv_values(moved / ".env")["CHAINLIT_AUTH_SECRET"]
        != dotenv_values(second / ".env")["CHAINLIT_AUTH_SECRET"]
    )
    monkeypatch.setenv("LOCALCHAT_CONFIG_DIR", str(moved))
    settings = load_settings()
    assert settings.data_dir == moved / ".local-agent-chat"
    assert (settings.data_dir / "existing.sqlite3").read_bytes() == b"persisted"
    assert not (second / ".local-agent-chat" / "existing.sqlite3").exists()


@pytest.mark.parametrize(
    "prefix,setting,expected",
    [
        ("/user/alice/", "auto", "/user/alice/vscode/proxy/8766"),
        ("/user/alice", "auto", "/user/alice/vscode/proxy/8766"),
        ("", "auto", ""),
        ("/user/alice/", "", ""),
        ("/user/alice/", "/custom/", "/custom"),
        (
            "/user/alice/",
            "${JUPYTERHUB_SERVICE_PREFIX%/}/vscode/proxy/$APP_PORT",
            "/user/alice/vscode/proxy/8766",
        ),
        (
            "",
            "${JUPYTERHUB_SERVICE_PREFIX%/}/vscode/proxy/$APP_PORT",
            "/vscode/proxy/8766",
        ),
    ],
)
def test_proxy_prefix_uses_final_port(monkeypatch, prefix, setting, expected):
    from local_agent_chat.cli import resolve_root_path

    monkeypatch.setenv("JUPYTERHUB_SERVICE_PREFIX", prefix)
    assert resolve_root_path(setting, 8766) == expected


def test_port_reservation_skips_busy_ports_and_keeps_selected_port_bound():
    import socket

    from local_agent_chat.cli import reserve_port

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        start = busy.getsockname()[1]
        with reserve_port("127.0.0.1", start) as first:
            assert first.getsockname()[1] > start
            with reserve_port("127.0.0.1", start) as second:
                assert second.getsockname()[1] > first.getsockname()[1]


@pytest.mark.parametrize("error_number", [13, 98])
def test_port_errors_and_upper_bound(monkeypatch, error_number):
    import errno

    from local_agent_chat import cli

    error_number = errno.EACCES if error_number == 13 else errno.EADDRINUSE
    attempted = []

    class Socket:
        def setsockopt(self, *args):
            pass

        def bind(self, address):
            attempted.append(address)
            raise OSError(error_number, "test bind error")

        def close(self):
            pass

    monkeypatch.setattr(cli.socket, "socket", lambda *args: Socket())
    with pytest.raises(PermissionError if error_number == errno.EACCES else ValueError):
        cli.reserve_port("127.0.0.1", 65535)
    assert attempted == [("127.0.0.1", 65535)]
