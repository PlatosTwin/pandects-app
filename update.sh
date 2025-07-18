#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_PUBLIC_DIR="$PROJECT_ROOT/frontend/public"
# adjust this if your generator outputs a different filename
OPENAPI_FILENAME="openapi.yaml"

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
python -m flask gen-openapi

# ── Move spec to frontend ─────────────────────────────────────────────────────
echo "📦 Moving $OPENAPI_FILENAME to $FRONTEND_PUBLIC_DIR..."
if [[ ! -f "$OPENAPI_FILENAME" ]]; then
  echo "Error: $OPENAPI_FILENAME not found in $BACKEND_DIR" >&2
  exit 1
fi
mv -f "$OPENAPI_FILENAME" "$FRONTEND_PUBLIC_DIR/"

# ── Freeze backend requirements ───────────────────────────────────────────────
echo "🐍 Freezing backend requirements to requirements.txt..."
# Use the venv’s pip to freeze
python -m pip freeze > requirements.txt

# ── Done ─────────────────────────────────────────────────────────────────────
echo "✅ Done! OpenAPI spec moved and requirements.txt updated."
