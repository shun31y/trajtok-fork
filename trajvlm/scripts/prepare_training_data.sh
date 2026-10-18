#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/scripts/data_env.sh"
# Stage1 downloads (including HF caches/logs) go entirely to the PixMo disk.
# Stage2 and model-asset paths remain under the main MOLMO_DATA_DIR.
if [[ "${1:-}" == "pixmo_cap" ]]; then
  export MOLMO_DATA_DIR="$PIXMO_DATA_DIR"
  export PIXMO_IMAGE_DIR="$PIXMO_DATA_DIR/torch_datasets/pixmo_images"
  export HF_HOME="$PIXMO_DATA_DIR/huggingface"
  export HF_DATASETS_CACHE="$PIXMO_DATA_DIR/hf_datasets"
fi
cd "$HERE"
exec uv run --locked --extra train python scripts/prepare_training_data.py "$@"
