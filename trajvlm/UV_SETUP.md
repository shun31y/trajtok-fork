# Reproduce the TrajVLM environment and data setup

Run from the repository root on Linux x86_64 (tested with uv 0.11.26):

```bash
uv python install 3.12.13
uv sync --project trajvlm --locked --extra train
source trajvlm/.venv/bin/activate
source trajvlm/scripts/data_env.sh
```

`trajvlm/.python-version` fixes Python; `trajvlm/uv.lock` fixes the Python
packages, including PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128,
torchcodec 0.9.0 and Transformers 4.57.6. The segmenter is an editable
relative-path dependency, so clone the whole repository. The environment
itself is excluded from Git. `--extra all` additionally installs development
and serving packages; it is not necessary for this setup.

## Data layout

The main root for Stage2 video data and model assets is
`/mnt/nas5/shun/VideoUnderstanding/trajtok` (`MOLMO_DATA_DIR`).
PixMo data use `/mnt/ssd2/shun/VideoUnderstanding/trajtok` (`PIXMO_DATA_DIR`).
On another machine, set these variables before sourcing `data_env.sh`.
For PixMo rows below, "root" means `PIXMO_DATA_DIR`; the other data rows use
`MOLMO_DATA_DIR`. HF caches and reports for each download use its stage's root.

| Path under the root | Contents |
| --- | --- |
| `huggingface/` | Hugging Face hub cache |
| `hf_datasets/` | Hugging Face datasets cache |
| `torch_datasets/pixmo_images/` | PixMoCap image files fetched from source URLs |
| `torch_datasets/pixmo_datasets/cap/` | Prepared PixMoCap train/validation data |
| `video_datasets/` | Stage2 academic video datasets and annotations |
| `setup_reports/` | Per-dataset download results |

```bash
bash trajvlm/scripts/prepare_training_data.sh pixmo_cap --n-procs 16
bash trajvlm/scripts/prepare_training_data.sh video_academic --n-procs 8
```

The Stage2 list is read directly from `VIDEO_ACADEMIC_DATASETS` in
`launch_scripts/sft.py` (34 mixture names, including shared download classes).
List it without downloading:

```bash
bash trajvlm/scripts/prepare_training_data.sh video_academic --list
```

These commands reuse the upstream caches. Keep the root path stable: prepared
PixMoCap records contain absolute image paths. The script continues after
individual download errors, writes a JSON report, and exits nonzero if any
failed. Methods that only download annotations can return successfully while
videos are still missing. `download_method_returned` deliberately does not mean
that the dataset has been validated for training. PixMoCap source URLs may have
expired; the upstream downloader filters unavailable images.

Many Stage2 datasets require manually obtained videos or approved access
(e.g. ActivityNet, COIN, Ego4D, Kinetics, Moments in Time, YouCook2).
Consult the error in `setup_reports/video_academic.json` and each dataset's
`download()` implementation in `olmo/data/academic_video_datasets.py` or
`olmo/data/molmo2_datasets.py` for the exact target directory. Downloading
`video_academic` alone does not prepare the entire default Molmo2 SFT mixture.

## Validation and remaining training prerequisites

```bash
uv pip check --python trajvlm/.venv/bin/python
cd trajvlm
uv run --locked --extra train python -c 'from launch_scripts import trajvlm_pretrain, trajvlm_sft; print("imports OK")'
```

This setup installs the Python training dependencies and imports the training
entrypoints. It does not launch training or download model weights. Before
training, prepare the segmenter/SigLIP2/Qwen3 weights, the DINOv3 source tree,
and any additional mixture data required by the chosen launch configuration.
The meta-safe DINOv3 wrapper uses `DINOV3_SOURCE_DIR`, defaulting to
`$MOLMO_DATA_DIR/dinov3`, for the official source checkout. The released full
segmenter checkpoint supplies its ConvNeXt weights; no separate DINOv3
weight download is needed for these TrajVLM recipes.

On the initial preparation host, `nvidia-smi` could not communicate with the
NVIDIA driver and `torch.cuda.is_available()` was false. GPU execution is
therefore unverified. The system FFmpeg is 4.2; the Dockerfile specifies FFmpeg
6 or 7 for TorchCodec. A working NVIDIA driver and compatible FFmpeg shared
libraries are host requirements, outside the uv lock. Optional compiled CUDA
extensions (flash-attn, ring-flash-attn, grouped-gemm) from the Dockerfile are
not installed by this uv environment and need to be prepared for the selected
training hardware/configuration.

## YAML experiment management

From the repository root, activate the data environment and preview Stage1:

```bash
source trajvlm/scripts/data_env.sh
uv run --project trajvlm --locked --extra train python \
  trajvlm/launch_scripts/trajvlm_experiment.py \
  --config trajvlm/configs/experiments/stage1_pixmo_cap.yaml --print-config
```

`--print-config` validates and prints the composed recipe plus resolved
`TrainConfig`. `--export-config /tmp/experiment.yaml` exports the same document;
both modes exit before process-group/CUDA/model/data/weight/W&B initialization.
They do not create the training output directory. They can run while data is
still downloading. The report distinguishes static validation from unverified
runtime readiness and distributed batch checks when world size is unknown.
This is separate from the trainer's `dry_run`, which constructs a model.

Recipes merge `extends` bases in order, then the experiment, then repeated
`--set key=value` options in order. Mappings merge recursively; lists replace.
Unknown keys, invalid types, cyclic inheritance and missing values fail early.
Relative paths use the YAML file that declared the value, independent of cwd;
paths set through `--set` use the top-level experiment YAML's directory.
Environment-based roots such as `MOLMO_DATA_DIR` must be absolute. URI values
are preserved. Run outputs and Stage2 checkpoint metadata currently require
local filesystem paths (NAS mounts are supported).

Create a derived YAML like `stage2_videoqa_lr_ablation.yaml`, or override a
single experiment value:

```bash
uv run --project trajvlm --locked --extra train python \
  trajvlm/launch_scripts/trajvlm_experiment.py \
  --config trajvlm/configs/experiments/stage1_pixmo_cap.yaml \
  --set run.name=stage1_lr1e4 --set run.output_dir=../../results/stage1_lr1e4 \
  --set training.connector_learning_rate=1e-4 --print-config
```

Stage1 is PixMoCap-only: it does not add pointing or Tulu data. Its fixed
22,347-step budget is the original run's approximate four-caption-epoch
reference, not four epochs of whichever images remain downloadable today.
To budget epochs using your actual prepared training split count:

```text
--set training.max_steps=null --set training.epochs=4 --set data.train_examples=YOUR_COUNT
```

The duration is epochs × ceil(train_examples / final global_batch_size).
Do not specify both max_steps and epochs. Dataset bodies are never read to
infer this count in the new recipe path. The old CLI retains its automatic
PixMoCap count behavior.

Once model assets, GPU drivers, kernels and data are prepared, launch Stage1
on one GPU or a node of GPUs (adjust batch sizes for the hardware):

```bash
uv run --project trajvlm --locked --extra train torchrun --nproc-per-node=1 \
  trajvlm/launch_scripts/trajvlm_experiment.py \
  --config trajvlm/configs/experiments/stage1_pixmo_cap.yaml

uv run --project trajvlm --locked --extra train torchrun --nproc-per-node=8 \
  trajvlm/launch_scripts/trajvlm_experiment.py \
  --config trajvlm/configs/experiments/stage1_pixmo_cap.yaml
```

For multiple nodes, use torchrun's `--nnodes`, `--node-rank`, `--master-addr`
and `--master-port` flags. Topology stays in torchrun and is recorded in the
run manifest. The global batch must be divisible by data-parallel size ×
micro batch; context-parallel size is accounted for when enabled in Stage2.
These commands have not been GPU-tested on the preparation host.

Stage2 requires an exact prepared Stage1 checkpoint directory containing
`config.yaml`, rather than a run root with several steps. It reads that model
metadata during preview, and loads weights only when training begins:

```bash
uv run --project trajvlm --locked --extra train python \
  trajvlm/launch_scripts/trajvlm_experiment.py \
  --config trajvlm/configs/experiments/stage2_videoqa.yaml \
  --set checkpoint=/absolute/path/to/stage1/step22347 --print-config
```

For a CPU-only demonstration before Stage1 exists, substitute
`--set checkpoint=../../tests/fixtures/pretrain_checkpoint`. This fixture has
no weights and must not be used to launch training. Add `torchrun
--nproc-per-node=8` instead of `python` and remove `--print-config` for a real
Stage2 run with an actual checkpoint.

Stage2 uses only `VIDEO_ACADEMIC_DATASETS`, not the full Molmo2 SFT mixture.
Set `--set 'data.datasets=[tgif,nextqa_mc]'` to select a prepared subset.
The sample recipe uses 64 frames and 8 frames_per_clip, matching `sft.sh`;
legacy Python SFT defaults remain 128/16. Evaluation and W&B are explicitly
disabled in both baseline recipes. To enable them, supply evaluation targets
and intervals, or tracking project/entity, respectively. `evaluation.tasks`
selects inference tasks and `evaluation.loss_tasks` selects loss tasks.

`trainer_overrides` allows only save_interval, save_num_checkpoints_to_keep,
max_grad_norm, console_log_interval, softmax_auxiliary_loss and
softmax_auxiliary_loss_scale. Model/data/batch/duration/optimizer changes must
use public recipe fields, so dependent settings are generated consistently.
The existing pretrain.py/SFT CLI wrappers retain their previous options.

New runs require a fresh output directory. Under it, the trainer retains its
normal `config.yaml` and checkpoints. `experiment/` stores:

- `recipe.yaml` and `train_config.yaml`: resolved experiment and trainer settings.
- `inputs/`: snapshots of every source YAML in composition order.
- `manifest.json`: Git SHA/dirty state, uv.lock SHA256, Python/package versions,
  source hashes, explicit overrides, command and distributed topology.

Environment variables unrelated to the recipe, including credentials, are
not captured. To resume an experiment, use the same recipe/output directory
with `--set run.resume=true`. An existing manifest, `config.yaml` and resumable
checkpoint are required. Each resume writes a separate `experiment/resumes/`
record, preserves the original provenance and keeps the trainer's configuration
comparison behavior. A run that failed before its first checkpoint cannot be
resumed; use a fresh output directory for a retry.

Run the CPU contract/compatibility tests without installing development extras:

```bash
uv run --project trajvlm --locked --extra train python -m unittest discover \
  -s trajvlm/tests -p test_experiment_config.py -v
```

## Stage-specific PEFT

The standard YAML recipes now enable these trainable scopes:

| Stage | Trainable | Frozen |
| --- | --- | --- |
| Stage1 (`peft.mode: connector_only`) | `vision_backbone.cross_attn` (TrajPerceiver), `image_projector` | LLM, ViT, segmenter, `query_proj`, embeddings and LM head |
| Stage2 (`peft.mode: connector_lora`) | Same TrajPerceiver/Projector plus LLM LoRA A/B | All base LLM weights and every other module |

`query_proj` is the separate 512→1152 bridge and is frozen. The Projector in
this policy is `image_projector`, which maps trajectory tokens to LLM features.
Learning rates, data, batch, duration, scheduler, FSDP and evaluation settings
are unchanged. LoRA parameters use the existing LLM optimizer group/rate.

Stage2 defaults are:

```yaml
peft:
  mode: connector_lora
  lora:
    rank: 16
    alpha: 32
    dropout: 0.0
    target_modules: [att_proj, attn_out]
```

The custom Molmo LLM uses fused Q/K/V in `att_proj`, so these are attention
input/output projections. This is native LoRA for the repository's LLM, not a
Hugging Face PEFT wrapper, and adds no dependency. Original Linear weight keys
remain intact. Configurations using separate attention projections can explicitly
choose `q_proj`, `k_proj`, `v_proj`, `attn_out`; unmatched targets fail during
model construction. Adapter settings can be overridden with e.g.
`--set peft.lora.rank=8 --set peft.lora.alpha=16` and appear in previews and
saved model configuration.

Stage1→Stage2 initializes new adapters with zero initial delta, while loading
all Stage1 base/connector weights. Full-model checkpoints continue to include
both base weights and adapters. Resuming Stage2 requires the same PEFT/LoRA
configuration and restores adapter and optimizer state; missing base weights
or missing adapters on resume fail. Both sharded and unsharded checkpoints are
supported. Changing rank/targets/alpha/dropout while resuming is rejected.

The PEFT defaults apply to the YAML recipes. Legacy CLI launchers retain their
PEFT-unset behavior; use the YAML launcher for the staged policy. To explicitly
recover the previous full-finetuning YAML behavior, use `--set peft=null`.

The CPU tests check exact optimizer membership, frozen parameters, actual
backward/update behavior, the real custom Qwen block's adapter construction,
meta materialization and checkpoint transition/resume with Gloo:

```bash
uv run --project trajvlm --locked --extra train python -m unittest discover \
  -s trajvlm/tests -p test_peft.py -v
```

The checkpoint tests require permission to create local loopback sockets.
GPU/FSDP execution is still unverified on this host.

## Qwen2.5-7B-Instruct as the base LLM

Choose the model in a Stage1 recipe:

```yaml
model:
  llm: qwen2.5_7b_instruct
```

Or pass `--set model.llm=qwen2.5_7b_instruct` to the YAML launcher. Dedicated
recipes are available as `stage1_pixmo_cap_qwen25_7b.yaml` and
`stage2_videoqa_qwen25_7b.yaml`. Existing default recipes continue to use
Qwen3-4B-Instruct. Data, batch size, learning rates, duration and PEFT settings
are unchanged by the new recipes.

Before a real training run, prepare the Qwen2.5 weights with:

```bash
source trajvlm/scripts/data_env.sh
uv run --project trajvlm --locked --extra train python \
  trajvlm/scripts/prepare_pretrained_model.py qwen2.5_7b_instruct --unsharded
```

This command downloads the actual 7B model from `Qwen/Qwen2.5-7B-Instruct`
and converts its QKV biases, MLP and independent output head to native format.
It writes `$MOLMO_DATA_DIR/pretrained_llms/qwen2.5-7b-instruct.pt`.
`--unsharded` is required here: the LLM initialization path expects a .pt file.
The conversion uses float32 host weights and requires sufficient CPU memory
and storage. The model weights have not been downloaded by this change.

Preview Stage1 without those weights:

```bash
uv run --project trajvlm --locked --extra train python \
  trajvlm/launch_scripts/trajvlm_experiment.py \
  --config trajvlm/configs/experiments/stage1_pixmo_cap_qwen25_7b.yaml --print-config
```

For Stage2, use `stage2_videoqa_qwen25_7b.yaml` with the checkpoint from the
Qwen2.5 Stage1 run, overriding `checkpoint` if the path or step differs.
Stage2 inherits the LLM architecture, tokenizer and projector output dimension
from that checkpoint; do not set `model.llm` again or substitute a Qwen3
checkpoint. The existing `message_format: qwen3` label denotes the common
ChatML user/assistant boundary format and is also used for this Qwen2.5 model.
Its tokenizer is correctly selected as `Qwen/Qwen2.5-7B-Instruct`.

CPU tests use a small randomly initialized HF Qwen2 model (no downloaded
weights), compare converted logits numerically, and test pretrained loading,
LoRA and configuration inheritance:

```bash
uv run --project trajvlm --locked --extra train python -m unittest discover \
  -s trajvlm/tests -p test_qwen25.py -v
```

Actual 7B-weight conversion and GPU training remain unverified.


## PixMoCap on SSD, Stage2 on NAS

`source scripts/data_env.sh` exports separate roots:

```bash
export MOLMO_DATA_DIR=/mnt/nas5/shun/VideoUnderstanding/trajtok
export PIXMO_DATA_DIR=/mnt/ssd2/shun/VideoUnderstanding/trajtok
```

The PixMo downloader routes its images, prepared records, HF caches and reports
to the SSD root. Stage2 continues to use the NAS root. For training, keep the
main root pointing to the existing model assets and set `PIXMO_DATA_DIR` to the
SSD; changing the base-model/VideoQA paths is unnecessary. All PixMo dataset
classes respect the optional PixMo root; when it is unset, their legacy
`MOLMO_DATA_DIR` layout remains supported. An explicit `PIXMO_IMAGE_DIR` can
still override image loading, while the PixMo download wrapper deliberately
sets that directory under the selected SSD root.

```bash
bash trajvlm/scripts/prepare_training_data.sh pixmo_cap --n-procs 16
bash trajvlm/scripts/prepare_training_data.sh video_academic --n-procs 8
```

Do not launch a second PixMo download while the existing SSD job is running.
For the initial migration, the NAS PixMo process and all child workers were
stopped; the NAS cache is retained. A detached migration job copies valid
images and the PixMo HF metadata to SSD, quarantines old failure-text cache
records, then starts the SSD downloader. Its status and output are in:

```text
/mnt/ssd2/shun/VideoUnderstanding/trajtok/setup_reports/migration.json
/mnt/ssd2/shun/VideoUnderstanding/trajtok/logs/pixmo_migration.log
/mnt/ssd2/shun/VideoUnderstanding/trajtok/logs/pixmo_cap_download.log
```

The last log is created after migration completes. Prepared PixMo records are
built with SSD image paths; retain this path or regenerate the records if the
image root changes again. NAS data are not deleted by this migration.

## 100k PixMoCap + Qwen2.5-7B-Instruct LoRA pretraining

From the repository root:

```bash
source trajvlm/configs/experiments/stage1_pixmo_cap_100k.env.sh
export WANDB_PROJECT=trajvlm
export WANDB_ENTITY=YOUR_WANDB_USER_OR_TEAM
trajvlm/.venv/bin/python trajvlm/launch_scripts/trajvlm_experiment.py \
  --config trajvlm/configs/experiments/stage1_pixmo_cap_100k_qwen25_7b_lora.yaml \
  --print-config
```

This recipe trains the connector and LLM attention LoRA (rank 16, alpha 32,
no dropout), freezing the ViT and base LLM. It uses global batch 128,
microbatch 1, and 3128 steps (four passes of 782 batches for 100000 examples).
Learning rates and warmup settings inherit the pretraining base: 2e-4 for
the connector with 200 warmup steps, and 2e-5 for LoRA with 2000 warmup
steps. Checkpoints are saved every 782 steps, retaining four. Evaluation runs
every 782 steps on 2000 held-out examples, reporting caption-only loss
(`pixmo_cap`) and caption/transcript loss (`pixmo_cap_with_transcripts`) to W&B.

Before training, prepare a Hugging Face DatasetDict at `$PIXMO_CAP_DATASET`
with a `train` split containing 100000 matching image records and a
`validation` split containing 2000 additional records. Preserve the
PixMoCap `caption`, `transcripts`, `image_url` and `image` columns; `image`
training paths must point into `$PIXMO_IMAGE_DIR` (`pixmo_images_100k`).
Validation paths point into the separate `pixmo_images_eval2k` directory.
`PIXMO_CAP_DATASET` overrides the prepared PixMoCap dataset location only;
`PIXMO_IMAGE_DIR` alone does not filter or rewrite prepared records.
This experiment declares `data.pixmo_cap_path` in YAML. The launcher uses
that path for training and evaluation, overriding `PIXMO_CAP_DATASET` in
the parent, torchrun workers and DataLoader workers. To change this recipe's
dataset, edit that field or use `--set data.pixmo_cap_path=/absolute/path`.
`data.train_examples` sets the training budget and does not cap dataset size.
Preparing these records and launching GPU training are separate steps.
Existing model weights and the segmenter checkpoint are also required.

W&B is enabled by default for this recipe. Set `WANDB_PROJECT` and
`WANDB_ENTITY` to your project and user/team, and authenticate W&B before
training. The run name matches `run.name`. To disable tracking, override
all three fields: `--set tracking.enabled=false --set tracking.project=null
--set tracking.entity=null`.

GPU selection for this experiment is configured in YAML:

```yaml
launch:
  gpus: [0]
```

After sourcing the environment and configuring W&B as above, launch from the
repository root with:

```bash
mkdir -p trajvlm/logs
set -o pipefail
trajvlm/.venv/bin/python trajvlm/launch_scripts/trajvlm_experiment.py \
  --config trajvlm/configs/experiments/stage1_pixmo_cap_100k_qwen25_7b_lora.yaml \
  --launch 2>&1 | tee "trajvlm/logs/stage1_pixmo_cap_100k_qwen25_7b_lora_$(date +%Y%m%d_%H%M%S).log"
```

`--launch` sets `CUDA_VISIBLE_DEVICES` to the configured IDs and starts one
worker per GPU with single-node standalone torchrun. IDs refer to devices
visible to the host/container before this mask; this replaces any inherited
`CUDA_VISIBLE_DEVICES`. The global batch must be divisible by GPU count times
microbatch size. Use `--set 'launch.gpus=[0,1]'` for a temporary override.
`--print-config` and `--export-config` inspect configuration without launching
workers, even with `--launch`. Existing direct torchrun commands remain
supported and control device selection themselves; `launch.gpus` applies to
`--launch` only.

## Preparing DINOv3 and the existing 100k image subset

The source checkout prepared on this host is:

```text
$MOLMO_DATA_DIR/dinov3
facebookresearch/dinov3 commit 6876159a11b4df116f30f667f8c9888617df0751
```

For another host, clone the official source and pin the same revision:

```bash
git clone https://github.com/facebookresearch/dinov3.git "$MOLMO_DATA_DIR/dinov3"
git -C "$MOLMO_DATA_DIR/dinov3" checkout 6876159a11b4df116f30f667f8c9888617df0751
```

The wrapper imports only the backbone factories, avoiding optional segmentation
evaluation dependencies. The segmenter loader extracts `vision_encoder.*`
from the released checkpoint, validates every required key and shape, and
supports plain tensors and distributed DTensors.
The loader preserves shared LayerNorm aliases (`norm` / `norms.3`) by
broadcasting and converting each tensor to the destination mesh/placement
before copying. CPU tests cover one and two ranks with shared parameters;
the actual vision backbone's 400 segmenter state entries were also verified
against the released checkpoint under single-GPU FSDP on this host.

To reproduce the 100k prepared records from the existing SSD metadata cache:

```bash
source trajvlm/configs/experiments/stage1_pixmo_cap_100k.env.sh
trajvlm/.venv/bin/python trajvlm/scripts/prepare_pixmo_subset.py \
  --metadata-dir "$PIXMO_DATA_DIR/hf_datasets/allenai___pixmo-cap/default/0.0.0/edce6390d9d5be6c8db0d863fbe62718c88988a4" \
  --image-dir "$PIXMO_IMAGE_DIR" \
  --output "$PIXMO_DATA_DIR/torch_datasets/pixmo_datasets_100k/cap"
```

This reads cached Arrow metadata without downloading anything, matches images
by SHA256 of their URL, verifies all image files with Pillow, and preserves
caption/transcripts. Duplicate URLs use the first metadata record in shard
order. This initial output has exactly 100000 train records; add the
validation split with the command below. The script refuses existing
outputs and publishes only after verification and saving complete.
`preparation.json` inside the output records the source, counts and duplicate
policy. It does not copy or alter the original image files.

Add 2000 evaluation images from the full PixMo image cache, excluding all
training URLs and filenames:

```bash
trajvlm/.venv/bin/python trajvlm/scripts/prepare_pixmo_validation.py \
  --train-dataset "$PIXMO_DATA_DIR/torch_datasets/pixmo_datasets_100k/cap" \
  --metadata-dir "$PIXMO_DATA_DIR/hf_datasets/allenai___pixmo-cap/default/0.0.0/edce6390d9d5be6c8db0d863fbe62718c88988a4" \
  --image-dir "$PIXMO_DATA_DIR/torch_datasets/pixmo_images" \
  --output "$PIXMO_CAP_DATASET" \
  --validation-images "$PIXMO_DATA_DIR/torch_datasets/pixmo_images_eval2k" \
  --count 2000
```

The configured dataset is `pixmo_datasets_100k/cap_with_validation` with
100000 train records and 2000 validation records. All validation images are
verified before copying. The original train-only dataset is retained.
Re-source the experiment environment file in existing shells to switch
`PIXMO_CAP_DATASET` to this combined dataset. No generation-based metrics
are enabled by this recipe.
