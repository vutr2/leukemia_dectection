#!/usr/bin/env bash
# One-time setup on macOS / Linux. Creates .venv and installs pinned packages.
set -euo pipefail
cd "$(dirname "$0")"
PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)'; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then echo "ERROR: Python 3.10-3.13 is required (3.12 recommended)." >&2; exit 1; fi
echo "Using $($PY --version)"
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-notebooks-lock.txt
.venv/bin/python run_local.py notebooks
.venv/bin/python run_local.py check || true
echo
echo "Setup complete. Next:  .venv/bin/python run_local.py v27"
