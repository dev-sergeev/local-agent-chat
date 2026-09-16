"""Configure and launch LocalChat without importing Chainlit before setup."""

from __future__ import annotations

import argparse
import errno
import getpass
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from dotenv.parser import parse_stream

from .installation import ASSETS, config_directory, data_directory, runtime_workspace
from .settings import AgentConfig, ModelProfile, parse_model

PROXY_TEMPLATE = "${JUPYTERHUB_SERVICE_PREFIX%/}/vscode/proxy/$APP_PORT"


def resolve_root_path(value: str, port: int) -> str:
    if value in {"auto", PROXY_TEMPLATE}:
        service_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "")
        value = (
            f"{service_prefix.rstrip('/')}/vscode/proxy/{port}"
            if service_prefix or value == PROXY_TEMPLATE
            else ""
        )
    prefix = "/" + value.strip("/") if value.strip("/") else ""
    if (
        any(char in prefix for char in ("?", "#", "\\", "$"))
        or any(char.isspace() or ord(char) < 32 for char in prefix)
        or any(part in {".", ".."} for part in prefix.split("/"))
    ):
        raise ValueError(
            "Root path must be auto or a URL path, such as /user/name/vscode/proxy/8765."
        )
    return prefix


def reserve_port(host: str, first_port: int) -> socket.socket:
    """Keep the selected port bound until the child inherits the socket."""
    for port in range(first_port, 65536):
        listener = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
            listener.listen(128)
            listener.setblocking(False)
            return listener
        except OSError as error:
            listener.close()
            if error.errno != errno.EADDRINUSE:
                raise
    raise ValueError(f"No free port from {first_port} to 65535.")


def _path(value: str | Path, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else relative_to / path).resolve()


def _endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("API base URL must be an http:// or https:// address.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "API base URL must not contain credentials, query or fragment."
        )
    return value.rstrip("/")


def _quote(value: str) -> str:
    if "\n" in value or "\r" in value or "\x00" in value:
        raise ValueError("Configuration values must be single-line text.")
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _write_private(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(content)


def initialize(args: argparse.Namespace) -> None:
    directory = _path(args.config_dir or config_directory(), Path.cwd())
    for name in (".env", "models.yaml"):
        if (directory / name).exists():
            raise ValueError(
                f"Configuration already exists in {directory}. Edit it to change settings; "
                "init never overwrites existing configuration."
            )
    defaults = yaml.safe_load((ASSETS / "examples/models.example.yaml").read_text())[
        "models"
    ][0]
    model = args.model
    base_url = args.base_url
    if not args.no_input:
        model = (
            model
            or input(f"Model [{defaults['model']}]: ").strip()
            or defaults["model"]
        )
        provider, _ = parse_model(model)
        default_url = (
            "https://api.giga.chat/v1"
            if provider == "gigachat"
            else "https://openrouter.ai/api/v1"
        )
        base_url = (
            base_url or input(f"API base URL [{default_url}]: ").strip() or default_url
        )
    if not model or not base_url:
        raise ValueError("For --no-input, supply --model and --base-url.")
    provider, model_id = parse_model(model)
    base_url = _endpoint(base_url)
    key_name = args.api_key_env or (
        "GIGACHAT_ACCESS_TOKEN" if provider == "gigachat" else "OPENAI_API_KEY"
    )
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key_name) or not key_name.endswith(
        ("KEY", "TOKEN")
    ):
        raise ValueError("Choose an API-key environment name such as OPENAI_API_KEY.")
    if args.no_api_key and provider != "openai":
        raise ValueError(
            "--no-api-key is only supported for OpenAI-compatible local APIs."
        )
    key = "not-required" if args.no_api_key else os.environ.get(key_name, "")
    if not key and not args.no_input:
        key = getpass.getpass(
            f"API key / access token (saved privately in {directory / '.env'}): "
        ).strip()
    if not key:
        raise ValueError(
            f"Set {key_name} or use --no-api-key for a local unauthenticated provider."
        )
    profile = {
        "id": "default",
        "label": model_id,
        "model": model,
        "base_url": base_url,
        "api_key_env": key_name,
        "streaming": args.streaming,
        "max_tokens": args.max_tokens,
    }
    ModelProfile(**profile, api_key=key)
    if provider == "openai" and urlsplit(base_url).hostname == "openrouter.ai":
        profile["summary_options"] = {"extra_body": {"reasoning": {"enabled": False}}}
    values = {
        "APP_HOST": "127.0.0.1",
        "APP_PORT": "8765",
        "APP_ROOT_PATH": "auto",
        "APP_DATA_DIR": args.data_dir
        or os.environ.get("APP_DATA_DIR")
        or ".local-agent-chat",
        "MODEL_PROFILES_FILE": "models.yaml",
        "CHAINLIT_AUTH_SECRET": secrets.token_urlsafe(48),
        key_name: key,
    }
    environment = "# LocalChat settings. Environment variables override these values.\n"
    environment += (
        "# Values are literal; shell commands and substitutions are not executed.\n"
    )
    environment += "".join(
        f"{name}={_quote(value)}\n" for name, value in values.items()
    )
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    _path(values["APP_DATA_DIR"], directory).mkdir(
        parents=True, exist_ok=True, mode=0o700
    )
    _write_private(
        directory / "models.yaml",
        yaml.safe_dump({"models": [profile]}, sort_keys=False, allow_unicode=True),
    )
    try:
        _write_private(directory / ".env", environment)
    except BaseException:
        (directory / "models.yaml").unlink()
        raise
    print(
        f"Configuration created: {directory}\nStart the UI: localchat run --config-dir {str(directory)!r}"
    )


def run(args: argparse.Namespace) -> int:
    if (
        os.name != "posix"
        or not hasattr(os, "O_NOFOLLOW")
        or os.open not in os.supports_dir_fd
    ):
        raise ValueError(
            "LocalChat requires POSIX dir_fd/O_NOFOLLOW support; use Linux or WSL2."
        )
    directory = _path(args.config_dir or config_directory(), Path.cwd())
    env_file = directory / ".env"
    if not env_file.is_file() and not os.environ.get("MODEL_PROFILES_FILE"):
        raise ValueError(f"No configuration in {directory}. Run localchat init first.")
    if env_file.is_file():
        file_values = {}
        with env_file.open(encoding="utf-8") as source:
            for binding in parse_stream(source):
                if binding.error:
                    raise ValueError(
                        f"Invalid .env syntax on line {binding.original.line} in {env_file}."
                    )
                if binding.key and binding.value is not None:
                    file_values[binding.key] = binding.value
        for name, value in file_values.items():
            os.environ.setdefault(name, value)
    os.environ["LOCALCHAT_CONFIG_DIR"] = str(directory)
    os.environ["MODEL_PROFILES_FILE"] = str(
        _path(os.environ.get("MODEL_PROFILES_FILE", "models.yaml"), directory)
    )
    os.environ["APP_DATA_DIR"] = str(
        _path(
            args.data_dir
            or os.environ.get("APP_DATA_DIR")
            or data_directory(directory),
            directory,
        )
    )
    host = args.host or os.environ.get("APP_HOST", "127.0.0.1")
    try:
        port = int(
            args.port if args.port is not None else os.environ.get("APP_PORT", "8765")
        )
    except ValueError as error:
        raise ValueError("Port must be an integer from 1 to 65535.") from error
    if not 1 <= port <= 65535:
        raise ValueError("Port must be an integer from 1 to 65535.")
    prefix = (
        args.root_path
        if args.root_path is not None
        else os.environ.get("APP_ROOT_PATH", "auto")
    )
    # Validate before creating runtime files; recalculate after reserving a port.
    os.environ["APP_ROOT_PATH"] = resolve_root_path(prefix, port)
    secret = os.environ.get("CHAINLIT_AUTH_SECRET", "")
    if len(secret) < 32 or secret.startswith("replace-"):
        raise ValueError(
            "CHAINLIT_AUTH_SECRET must contain a random secret of at least 32 characters. Run localchat init for a new configuration."
        )
    from .settings import load_settings

    settings = load_settings()
    for profile in settings.models:
        if not profile.api_key or profile.api_key.startswith("replace-"):
            raise ValueError(
                f"Set {profile.api_key_env} for model profile {profile.id}."
            )
    with (
        runtime_workspace(settings.data_dir) as root,
        reserve_port(host, port) as listener,
    ):
        port = listener.getsockname()[1]
        prefix = resolve_root_path(prefix, port)
        os.environ["APP_HOST"] = host
        os.environ["APP_PORT"] = str(port)
        os.environ["APP_ROOT_PATH"] = prefix
        os.environ["CHAINLIT_APP_ROOT"] = str(root)
        os.environ["CHAINLIT_ENV_FILE"] = os.devnull
        os.environ["CHAINLIT_HOST"] = host
        os.environ["CHAINLIT_PORT"] = str(port)
        os.environ["CHAINLIT_ROOT_PATH"] = prefix
        display_host = (
            "127.0.0.1" if host == "0.0.0.0" else (f"[{host}]" if ":" in host else host)
        )
        print(
            f"LocalChat: http://{display_host}:{port}{prefix}/\nConfiguration: {directory}\nData: {settings.data_dir}\nPress Ctrl+C to stop.",
            flush=True,
        )
        arguments = [
            sys.executable,
            "-m",
            "local_agent_chat.server",
            str(listener.fileno()),
        ]
        if args.open_browser:
            arguments.append("--open-browser")
        # Chainlit's lifespan calls os._exit(), bypassing Python finalizers.
        # Own its writable workspace in a parent process so cleanup still runs.
        # A separate process group prevents Ctrl+C being delivered twice.
        process = subprocess.Popen(
            arguments, start_new_session=True, pass_fds=(listener.fileno(),)
        )
        listener.close()

        def forward_signal(signum, _frame):
            if process.poll() is None:
                process.send_signal(signum)

        previous = {
            signum: signal.signal(signum, forward_signal)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            return process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="localchat",
        description=(
            "LocalChat: a local chat UI with a sandboxed ReAct agent. "
            "Supports OpenAI-compatible APIs and native GigaChat models. "
            "Model requests run sequentially; a new request is rejected while busy."
        ),
        epilog=(
            "GigaChat setup: localchat init --model gigachat:GigaChat-2 "
            "--base-url https://api.giga.chat/v1. Use a ready access token. "
            "New profiles default to streaming=false and max_tokens=128000."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('local-agent-chat')}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init",
        help="Configure OpenAI-compatible or GigaChat models and a session secret.",
        description=(
            "Create private settings for OpenAI-compatible APIs or native GigaChat. "
            "GigaChat example: --model gigachat:GigaChat-2 "
            "--base-url https://api.giga.chat/v1. "
            "Use GIGACHAT_ACCESS_TOKEN with a ready token, not OAuth credentials."
        ),
    )
    init.add_argument(
        "--config-dir", help="Configuration directory (default: current directory)."
    )
    init.add_argument(
        "--data-dir", help="Data directory (default: .local-agent-chat beside .env)."
    )
    init.add_argument(
        "--model", help="Model identifier: openai:<model-id> or gigachat:<model-id>."
    )
    init.add_argument(
        "--base-url", help="Provider's API base URL, including /v1 if required."
    )
    init.add_argument(
        "--api-key-env",
        help="Token variable (default: OPENAI_API_KEY or GIGACHAT_ACCESS_TOKEN).",
    )
    init.add_argument(
        "--no-api-key",
        action="store_true",
        help="Use a placeholder key for a local unauthenticated API.",
    )
    init.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable streamed responses (default: disabled; --no-streaming also supported).",
    )
    init.add_argument(
        "--max-tokens",
        type=int,
        default=AgentConfig().max_output_tokens,
        help="Maximum response tokens saved in the model profile (default: 128000).",
    )
    init.add_argument(
        "--no-input",
        action="store_true",
        help="Require settings through arguments/environment; never prompt.",
    )
    start = commands.add_parser(
        "run",
        help="Start the UI with saved OpenAI-compatible or GigaChat settings.",
        description=(
            "Run OpenAI-compatible or native GigaChat models from models.yaml. "
            "Requests are sequential across chats, summaries and titles. "
            "A new request is rejected while busy. Transient errors get up to "
            "10 retries with delays of 1, 2, 4, ... seconds, capped at 300 seconds. "
            "LLM_MAX_RETRIES overrides the retry count; 0 disables retries."
        ),
    )
    start.add_argument(
        "--config-dir", help="Directory containing .env and models.yaml."
    )
    start.add_argument("--data-dir", help="Override the persistent data directory.")
    start.add_argument("--host", help="Listening address (default: 127.0.0.1).")
    start.add_argument("--port", type=int, help="First port to try (default: 8765).")
    start.add_argument(
        "--root-path",
        help="Public URL prefix; auto detects JupyterHub, empty disables it.",
    )
    start.add_argument(
        "--open-browser", action="store_true", help="Open a browser on startup."
    )
    args = parser.parse_args(argv)
    try:
        status = (initialize if args.command == "init" else run)(args)
    except (ValueError, OSError, yaml.YAMLError) as error:
        print(f"localchat: {error}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        return 130
    return status or 0
