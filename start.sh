#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root relative to this script
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Activate local virtual environment if present
if [[ -d "$ROOT_DIR/.venv" ]]; then
    source "$ROOT_DIR/.venv/bin/activate"
fi

# Basic sanity check for API key file
if [[ ! -f "$ROOT_DIR/OPENROUTER_API_KEY.txt" ]]; then
    echo "Error: OPENROUTER_API_KEY.txt not found in $ROOT_DIR" >&2
    exit 1
fi

# Install/update dependencies unless skipped
if [[ "${SKIP_DEP_INSTALL:-0}" != "1" ]]; then
    python -m pip install --upgrade pip >/dev/null
    python -m pip install -r requirements.txt
fi

# Default Flask configuration for local dev
export FLASK_ENV=${FLASK_ENV:-development}
export FLASK_DEBUG=${FLASK_DEBUG:-1}
PORT=${PORT:-10000}

# Gracefully free the target port if another process is holding it
existing_pids=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
if [[ -n "$existing_pids" ]]; then
    echo "Stopping process using port $PORT..."
    kill $existing_pids 2>/dev/null || true
    sleep 1
    if lsof -ti tcp:"$PORT" >/dev/null 2>&1; then
        kill -9 $existing_pids 2>/dev/null || true
    fi
fi

# Start the Flask interface server
exec python interface.py
