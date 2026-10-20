# Run the 100k experiment from the shared NAS

The shared asset root is `/mnt/nas5/shun/VideoUnderstanding/trajtok`.
It contains:

```text
pixmo_images_100k/                         # 100000 training images
pixmo_images_eval2k/                       # 2000 validation images
torch_datasets/pixmo_datasets_100k/
  cap/                                    # train-only prepared records
  cap_with_validation/                    # train 100000 + validation 2000
checkpoints/segmenter_filteredmixdata_all.pth
pretrained_image_encoders/siglip2-so400m-14-384.pt
pretrained_llms/qwen2.5-7b-instruct.pt
dinov3/                                   # official source checkout
huggingface/                              # existing HF model/tokenizer cache
```

Use the supplied source snapshot to include the fixes made on
server8. The manifest next to `source.tar.gz` records its SHA256 and base Git
revision. The snapshot excludes environments, credentials, logs, and results.
Extract `source.tar.gz` into a local workspace; it creates `trajtok-fork/`.

From that repository root:

```bash
uv python install 3.12.13
uv sync --project trajvlm --locked --extra train
source trajvlm/.venv/bin/activate

export MOLMO_DATA_DIR=/mnt/nas5/shun/VideoUnderstanding/trajtok
export PIXMO_DATA_DIR="$MOLMO_DATA_DIR"
export DINOV3_SOURCE_DIR="$MOLMO_DATA_DIR/dinov3"
export HF_HOME="$MOLMO_DATA_DIR/huggingface"
source trajvlm/scripts/data_env.sh
export WANDB_PROJECT=trajvlm
export WANDB_ENTITY=YOUR_WANDB_USER_OR_TEAM
wandb login

NAS_SET=(
  --set run.name=stage1_pixmo_cap_100k_qwen25_7b_lora_nas
  --set run.output_dir=../../results/stage1_pixmo_cap_100k_qwen25_7b_lora_nas
  --set "data.pixmo_cap_path=$MOLMO_DATA_DIR/torch_datasets/pixmo_datasets_100k/cap_with_validation"
)

python trajvlm/launch_scripts/trajvlm_experiment.py \
  --config trajvlm/configs/experiments/stage1_pixmo_cap_100k_qwen25_7b_lora.yaml \
  "${NAS_SET[@]}" --print-config

mkdir -p trajvlm/logs
set -o pipefail
python trajvlm/launch_scripts/trajvlm_experiment.py \
  --config trajvlm/configs/experiments/stage1_pixmo_cap_100k_qwen25_7b_lora.yaml \
  "${NAS_SET[@]}" --launch 2>&1 | tee "trajvlm/logs/pretrain_nas_$(date +%Y%m%d_%H%M%S).log"
```

`NAS_SET` points the SSD recipe at the NAS dataset and a separate run name and
output directory; GPU selection and all training/evaluation settings are
inherited from the SSD recipe. Override GPU selection with
`--set 'launch.gpus=[0,1]'` if needed. Results are written to the new server's local
`trajvlm/results/stage1_pixmo_cap_100k_qwen25_7b_lora_nas/`; the initial model
weights on NAS are never overwritten by training. There were no saved trained
checkpoints at transfer time.

Prepared records use absolute NAS image paths. If another server mounts the
NAS under a different path, set `MOLMO_DATA_DIR` accordingly and generate a
local metadata copy (images are reused):

```bash
python trajvlm/scripts/relocate_pixmo_dataset.py \
  --source "$MOLMO_DATA_DIR/torch_datasets/pixmo_datasets_100k/cap_with_validation" \
  --output /tmp/trajvlm-pixmo-cap \
  --train-images "$MOLMO_DATA_DIR/pixmo_images_100k" \
  --validation-images "$MOLMO_DATA_DIR/pixmo_images_eval2k"
```

Then add `--set data.pixmo_cap_path=/tmp/trajvlm-pixmo-cap` after
`"${NAS_SET[@]}"` when launching; the later value wins.
The relocation tool refuses existing outputs. A working NVIDIA driver and
the host libraries described in `UV_SETUP.md` are required on the new server.
