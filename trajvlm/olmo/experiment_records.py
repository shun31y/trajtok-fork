"""Run provenance, kept separate from the trainer's checkpoint configuration."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from omegaconf import OmegaConf


def check_output(experiment):
    r = experiment.recipe
    out = Path(r.run.output_dir)
    if r.run.resume:
        if not (out / 'config.yaml').is_file() or not (out / 'experiment/manifest.json').is_file():
            raise ValueError('run.resume requires an existing config.yaml and experiment/manifest.json')
        from olmo.train.checkpointer import Checkpointer
        try:
            Checkpointer.latest_checkpoint(str(out))
        except FileNotFoundError as error:
            raise ValueError(f'run.resume requires a resumable checkpoint in {out}') from error
    elif out.exists():
        raise ValueError(f'Output already exists: {out}. Choose a new run.output_dir or explicitly resume.')


def write_run_record(experiment, cfg, command=None):
    """Rank zero only. Reserve new outputs atomically and never replace provenance."""
    r = experiment.recipe
    out = Path(r.run.output_dir)
    if r.run.resume:
        check_output(experiment)
        record_dir = out / 'experiment/resumes' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8])
        record_dir.mkdir(parents=True, exist_ok=False)
    else:
        out.mkdir(parents=True, exist_ok=False)
        record_dir = out / 'experiment'
        record_dir.mkdir()
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()

    try:
        revision, dirty = git('rev-parse', 'HEAD'), bool(git('status', '--porcelain'))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    versions = {}
    for package in ('torch', 'transformers', 'omegaconf', 'datasets', 'trajvlm'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    lock = root / 'trajvlm/uv.lock'
    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'run_name': r.run.name, 'resume': r.run.resume,
        'git_revision': revision, 'git_dirty': dirty,
        'uv_lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
        'python': platform.python_version(), 'packages': versions,
        'command': list(command if command is not None else sys.argv),
        'topology': {key: os.environ.get(key) for key in
                     ('WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT')},
        'sources': [{'path': src['path'], 'snapshot': f'inputs/{i:03d}.yaml',
                     'sha256': hashlib.sha256(src['content'].encode()).hexdigest()}
                    for i, src in enumerate(experiment.sources)],
        'overrides': experiment.overrides,
    }
    (record_dir / 'inputs').mkdir()
    for i, src in enumerate(experiment.sources):
        (record_dir / f'inputs/{i:03d}.yaml').write_text(src['content'])
    OmegaConf.save(OmegaConf.structured(r), record_dir / 'recipe.yaml', resolve=True)
    OmegaConf.save(OmegaConf.structured(cfg), record_dir / 'train_config.yaml', resolve=True)
    (record_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    return record_dir
