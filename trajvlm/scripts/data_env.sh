# Source from any directory. Override MOLMO_DATA_DIR for another machine.
export MOLMO_DATA_DIR="${MOLMO_DATA_DIR:-/mnt/nas5/shun/VideoUnderstanding/trajtok}"
export PIXMO_DATA_DIR="${PIXMO_DATA_DIR:-/mnt/ssd2/shun/VideoUnderstanding/trajtok}"
export HF_HOME="${HF_HOME:-$MOLMO_DATA_DIR/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$MOLMO_DATA_DIR/hf_datasets}"
