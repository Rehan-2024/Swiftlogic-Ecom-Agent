#!/usr/bin/env bash
# Thin wrapper for CI / developers — delegates to the Python check.
# Usage:
#   ./scripts/check_env_frozen.sh              # strict
#   ./scripts/check_env_frozen.sh --skip-missing-tag   # warn-only before tag exists
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
exec python3 "${SCRIPT_DIR}/check_env_frozen.py" "$@"
