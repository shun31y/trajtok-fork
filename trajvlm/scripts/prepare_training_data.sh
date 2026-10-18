#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/scripts/data_env.sh"
cd "$HERE"
exec uv run --locked --extra train python scripts/prepare_training_data.py "$@"
