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

The default root is `/mnt/nas5/shun/VideoUnderstanding/trajtok`. On another
machine, set `MOLMO_DATA_DIR` before sourcing `data_env.sh`.

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
The legacy meta-safe DINOv3 wrapper still contains author-local paths that
must be made portable during model preparation.

On the initial preparation host, `nvidia-smi` could not communicate with the
NVIDIA driver and `torch.cuda.is_available()` was false. GPU execution is
therefore unverified. The system FFmpeg is 4.2; the Dockerfile specifies FFmpeg
6 or 7 for TorchCodec. A working NVIDIA driver and compatible FFmpeg shared
libraries are host requirements, outside the uv lock. Optional compiled CUDA
extensions (flash-attn, ring-flash-attn, grouped-gemm) from the Dockerfile are
not installed by this uv environment and need to be prepared for the selected
training hardware/configuration.
