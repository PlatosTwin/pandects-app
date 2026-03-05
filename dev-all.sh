#!/usr/bin/env bash

set -euo pipefail

# Re-exec under caffeinate once so long-running dev services are not interrupted by sleep.
if [[ "${DEVALL_CAFFEINATED:-0}" != "1" ]] && command -v caffeinate >/dev/null 2>&1; then
  export DEVALL_CAFFEINATED=1
  exec caffeinate -i bash "${BASH_SOURCE[0]}" "$@"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGSTER_HOME_DIR="${DAGSTER_HOME:-$ROOT_DIR/etl/.dagster_home}"
DAGSTER_TMPDIR_DIR="${DAGSTER_TMPDIR:-$ROOT_DIR/etl/.tmp_dagster_runtime}"

cleanup() {
  pkill -P $$ >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

echo "Starting backend on port ${FLASK_RUN_PORT:-5113}..."
(
  cd "$ROOT_DIR"
  # Flask app module is backend.app; run from repo root so import paths match dev usage.
  source backend/venv/bin/activate
  FLASK_APP=backend.app FLASK_RUN_PORT="${FLASK_RUN_PORT:-5113}" flask run
) &

echo "Starting docs..."
(
  cd "$ROOT_DIR/docs"
  npm run dev:clean
) &

echo "Starting frontend..."
(
  cd "$ROOT_DIR/frontend"
  npm run dev
) &

echo "Starting Dagster (dg dev)..."
(
  cd "$ROOT_DIR/etl"
  source .venv/bin/activate
  export DAGSTER_HOME="$DAGSTER_HOME_DIR"
  export TMPDIR="$DAGSTER_TMPDIR_DIR"
  mkdir -p "$DAGSTER_HOME" "$TMPDIR"
  dg dev
) &

wait
