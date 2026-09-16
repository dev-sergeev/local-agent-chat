"""Run Chainlit on the listening socket reserved by the CLI parent.

Chainlit 2.11's CLI cannot accept an inherited socket. Keep its initialization
here and hand the socket directly to Uvicorn, avoiding a check/bind race.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path


def main() -> None:
    import uvicorn
    from chainlit.cli import (
        assert_app,
        config,
        ensure_jwt_secret,
        init_lc_cache,
        init_markdown,
        load_module,
    )

    config.run.host = os.environ["CHAINLIT_HOST"]
    config.run.port = int(os.environ["CHAINLIT_PORT"])
    config.run.root_path = os.environ["CHAINLIT_ROOT_PATH"]
    config.run.headless = "--open-browser" not in sys.argv[2:]
    config.run.module_name = str(Path(__file__).with_name("app.py"))
    config.run.ssl_cert = os.environ.get("CHAINLIT_SSL_CERT")
    config.run.ssl_key = os.environ.get("CHAINLIT_SSL_KEY")
    if bool(config.run.ssl_cert) != bool(config.run.ssl_key):
        raise ValueError(
            "Both CHAINLIT_SSL_CERT and CHAINLIT_SSL_KEY must be provided."
        )
    # Chainlit builds static routes using config.run.root_path at import time.
    from chainlit.server import app

    load_module(config.run.module_name)
    ensure_jwt_secret()
    assert_app()
    init_markdown(config.root)
    init_lc_cache()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=config.run.host,
            port=config.run.port,
            log_level="error",
            ws=os.environ.get("UVICORN_WS_PROTOCOL", "auto"),
            ws_per_message_deflate=os.environ.get(
                "UVICORN_WS_PER_MESSAGE_DEFLATE", "true"
            ).lower()
            in {"true", "1", "yes"},
            ssl_certfile=config.run.ssl_cert,
            ssl_keyfile=config.run.ssl_key,
        )
    )
    with socket.socket(fileno=int(sys.argv[1])) as listener:
        asyncio.run(server.serve(sockets=[listener]))


if __name__ == "__main__":
    main()
