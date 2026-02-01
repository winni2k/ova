#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OVA_PROFILE="${OVA_PROFILE:-default}"
OVA_HOST="${OVA_HOST:-127.0.0.1}"
OVA_PORT="${OVA_PORT:-5173}"

echo "Starting OVA Backend..."
echo "  Profile: $OVA_PROFILE"
echo "  Host: $OVA_HOST"
echo "  Port: $OVA_PORT"

exec uv run uvicorn ova.api:app --host "$OVA_HOST" --port "$OVA_PORT"
