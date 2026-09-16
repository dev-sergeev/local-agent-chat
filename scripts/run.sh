#!/usr/bin/env bash
set -euo pipefail

# The Python CLI reads .env as data and resolves the proxy after choosing a port.
# Retain ENV_FILE for existing callers, without sourcing shell code.
if [[ -n "${ENV_FILE:-}" ]]; then
  if [[ "$(basename -- "$ENV_FILE")" != ".env" ]]; then
    echo "ENV_FILE must name a .env file; use localchat run --config-dir for its directory." >&2
    exit 2
  fi
  exec python -m local_agent_chat run --config-dir "$(dirname -- "$ENV_FILE")" "$@"
fi
exec python -m local_agent_chat run "$@"
