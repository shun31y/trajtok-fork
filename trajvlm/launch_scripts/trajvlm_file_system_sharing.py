"""Run TrajVLM with PyTorch's file-system tensor sharing strategy."""

from pathlib import Path
import os
import sys

import torch.multiprocessing as multiprocessing

TRAJVLM_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_DIR = Path(__file__).resolve().parent / "trajvlm_sharing_bootstrap"
sys.path.insert(0, str(TRAJVLM_ROOT))

from launch_scripts.trajvlm_experiment import main


if __name__ == "__main__":
    # TrajVLM forces the spawn start method. Workers start fresh interpreters,
    # so the strategy set here would not be inherited by the workers.
    # sitecustomize sets it in each spawned interpreter before DataLoader runs.
    previous_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(BOOTSTRAP_DIR) + (
        os.pathsep + previous_pythonpath if previous_pythonpath else ""
    )
    multiprocessing.set_sharing_strategy("file_system")
    raise SystemExit(main())
