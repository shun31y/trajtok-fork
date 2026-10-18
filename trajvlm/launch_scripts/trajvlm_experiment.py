"""Compose and inspect YAML experiments, or execute them under torchrun."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Editable installation exposes olmo; the source-only launch_scripts package
# also needs the TrajVLM directory when invoked by absolute script path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from yaml import YAMLError
from olmo.experiment_config import build_experiment, load_experiment, validate_train_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--print-config', action='store_true')
    parser.add_argument('--export-config', type=Path)
    args = parser.parse_args(argv)
    try:
        experiment = load_experiment(args.config, args.set)
        cfg = build_experiment(experiment)
        world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else None
        validate_train_config(cfg, experiment.recipe, world_size)
        report = OmegaConf.create({
            'validation': {
                'static': 'passed',
                'distributed_batch': 'passed' if world_size is not None else 'unverified: world size unknown',
                'runtime': 'unverified: data, weights, CUDA, FFmpeg and kernels are not checked',
            },
            'recipe': OmegaConf.structured(experiment.recipe),
            'train_config': OmegaConf.structured(cfg),
        })
        if args.print_config or args.export_config is not None:
            if args.print_config:
                print(OmegaConf.to_yaml(report, resolve=True))
            if args.export_config is not None and int(os.environ.get('RANK', '0')) == 0:
                args.export_config.parent.mkdir(parents=True, exist_ok=True)
                OmegaConf.save(report, args.export_config, resolve=True)
            return 0
        from olmo.experiment_records import check_output, write_run_record
        check_output(experiment)
        from olmo.util import prepare_torchrun_environment
        import torch
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        prepare_torchrun_environment()
        import torch.distributed as dist
        validate_train_config(cfg, experiment.recipe, dist.get_world_size())
        # Send a rank-zero recording failure to every worker, avoiding a barrier
        # hang if the output was concurrently created or the disk is unwritable.
        outcome = [None]
        if dist.get_rank() == 0:
            try:
                write_run_record(experiment, cfg)
            except Exception as error:
                outcome[0] = f'Could not record experiment: {error}'
        dist.broadcast_object_list(outcome, src=0)
        if outcome[0] is not None:
            raise ValueError(outcome[0])
        from olmo.train.run_trainer import run_trainer
        run_trainer(cfg)
        return 0
    except (ValueError, OSError, KeyError, OmegaConfBaseException, YAMLError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
