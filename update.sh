#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_PUBLIC_DIR="$PROJECT_ROOT/frontend/public"
DOCS_DIR="$PROJECT_ROOT/docs"
# adjust this if your generator outputs a different filename
OPENAPI_FILENAME="openapi.yaml"
OPENAPI_BACKEND_PATH="$BACKEND_DIR/$OPENAPI_FILENAME"
OPENAPI_FRONTEND_PATH="$FRONTEND_PUBLIC_DIR/$OPENAPI_FILENAME"

# ── Generate OpenAPI spec ────────────────────────────────────────────────────
echo "🔧 Generating OpenAPI spec in $BACKEND_DIR..."
cd "$BACKEND_DIR"

# Activate virtual environment
if [[ -f "venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "venv/bin/activate"
else
  echo "Error: Virtual environment not found at $BACKEND_DIR/venv" >&2
  exit 1
fi

# Use the venv’s Python to run Flask
python -m flask --app app gen-openapi

# ── Move spec to frontend ─────────────────────────────────────────────────────
echo "📦 Syncing $OPENAPI_FILENAME to $FRONTEND_PUBLIC_DIR..."
if [[ ! -f "$OPENAPI_BACKEND_PATH" ]]; then
  echo "Error: $OPENAPI_FILENAME not found in $BACKEND_DIR" >&2
  exit 1
fi
cp -f "$OPENAPI_BACKEND_PATH" "$OPENAPI_FRONTEND_PATH"

# ── Freeze backend requirements ───────────────────────────────────────────────
echo "🐍 Freezing backend requirements to requirements.txt..."
# Use the venv’s pip to freeze
python -m pip freeze > requirements.txt

# ── Regenerate docs API artifacts ─────────────────────────────────────────────
echo "📚 Regenerating docs API artifacts in $DOCS_DIR..."
cd "$DOCS_DIR"

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm is required to run docs generation." >&2
  exit 1
fi

if ! command -v yarn >/dev/null 2>&1; then
  if command -v corepack >/dev/null 2>&1; then
    echo "🧶 Enabling Corepack so yarn is available for Docusaurus..."
    corepack enable
  fi
fi

if ! command -v yarn >/dev/null 2>&1; then
  echo "Error: yarn is required to run docs generation. Install it or enable Corepack." >&2
  exit 1
fi

npm run clean-api
npm run gen-api

# ── Done ─────────────────────────────────────────────────────────────────────
echo "✅ Done! OpenAPI spec generated and synced, requirements.txt updated, and docs API artifacts cleaned + regenerated."
