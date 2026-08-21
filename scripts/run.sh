#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${ENV_FILE:=$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${CHAINLIT_AUTH_SECRET:?Set CHAINLIT_AUTH_SECRET to a random value of at least 32 bytes}"
if (( ${#CHAINLIT_AUTH_SECRET} < 32 )); then
  echo "CHAINLIT_AUTH_SECRET must contain at least 32 characters" >&2
  exit 2
fi
: "${MODEL_PROFILES_FILE:=models.yaml}"
: "${APP_DATA_DIR:=.local-agent-chat}"
: "${APP_ROOT_PATH:=}"
: "${APP_PORT:=8765}"

export CHAINLIT_AUTH_SECRET MODEL_PROFILES_FILE APP_DATA_DIR APP_ROOT_PATH APP_PORT
cd "$PROJECT_ROOT"
exec chainlit run app.py --host 0.0.0.0 --port "$APP_PORT" --root-path "$APP_ROOT_PATH"
