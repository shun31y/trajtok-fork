"""CPU contract tests; no downloaded datasets, weights or working GPU needed."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from olmo.experiment_config import build_experiment, load_experiment, validate_train_config
from olmo.experiment_records import check_output, write_run_record
from launch_scripts import trajvlm_pretrain as pretrain, trajvlm_sft as sft
from launch_scripts.trajvlm_experiment import main
from olmo.models.trajvlm_config import TrajVlmConfig
from olmo.train.trainer_config import TrainConfig

FIXTURES = ROOT / 'tests/fixtures'
STAGE1 = ROOT / 'configs/experiments/stage1_pixmo_cap.yaml'
STAGE2 = ROOT / 'configs/experiments/stage2_videoqa.yaml'


def serial(cfg):
    value = json.loads(json.dumps(OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True), default=str))
    # The new optional architecture field is absent from pre-PEFT snapshots.
    if value.get('model', {}).get('llm', {}).get('lora') is None:
        value.get('model', {}).get('llm', {}).pop('lora', None)
    return value


class Experiments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, MOLMO_DATA_DIR='/tmp/trajvlm-test-data',
                              WANDB_PROJECT='test-project', WANDB_ENTITY='test-entity')
        self.env.start()
        self.addCleanup(self.env.stop)

    def experiment(self, stage=1, overrides=()):
        options = [f'run.output_dir={self.root}/results', *overrides]
        if stage == 2:
            options.insert(0, f'checkpoint={FIXTURES}/pretrain_checkpoint')
        return load_experiment(STAGE1 if stage == 1 else STAGE2, options)

    def test_stage1_cpu_and_roundtrip(self):
        exp = self.experiment()
        cfg = build_experiment(exp)
        self.assertIsInstance(cfg.model, TrajVlmConfig)
        self.assertEqual(cfg.data.dataset, 'pixmo_cap_with_transcripts')
        self.assertIsNone(cfg.data.kwargs_mixture)
        self.assertIsNone(cfg.data.mixture)
        self.assertEqual(cfg.evaluators, [])
        self.assertEqual(cfg.inf_evaluators, [])
        self.assertIsNone(cfg.wandb)
        self.assertIsNone(cfg.compile)
        self.assertEqual(cfg.max_duration, 22347)
        self.assertFalse(Path(cfg.save_folder).exists())
        path = self.root / 'train.yaml'
        OmegaConf.save(OmegaConf.structured(cfg), path, resolve=True)
        restored = TrainConfig.load(path)
        self.assertIsInstance(restored.model, TrajVlmConfig)
        self.assertEqual(serial(restored), serial(cfg))

    def test_100k_qwen25_lora_pretrain(self):
        path = ROOT / 'configs/experiments/stage1_pixmo_cap_100k_qwen25_7b_lora.yaml'
        cfg = build_experiment(load_experiment(path))
        self.assertEqual(cfg.max_duration, 2084)
        self.assertEqual(cfg.stop_at, 2084)
        self.assertTrue(cfg.ft_connector)
        self.assertTrue(cfg.ft_llm)
        self.assertFalse(cfg.ft_vit)
        self.assertEqual(cfg.model.llm.lora.rank, 16)
        self.assertEqual(cfg.model.llm.lora.target_modules, ['att_proj', 'attn_out'])
        self.assertEqual(cfg.save_interval, 521)
        self.assertEqual(cfg.device_train_microbatch_size, 1)
        self.assertIsNotNone(cfg.wandb)
        self.assertEqual(cfg.wandb.project, 'test-project')
        self.assertEqual(cfg.wandb.entity, 'test-entity')
        self.assertEqual(cfg.eval_interval, 521)
        self.assertEqual(len(cfg.evaluators), 2)
        self.assertEqual({ev.data.dataset for ev in cfg.evaluators},
                         {'pixmo_cap', 'pixmo_cap_with_transcripts'})
        for ev in cfg.evaluators:
            self.assertEqual(ev.data.split, 'validation')
            self.assertEqual(ev.max_examples, 2000)
        self.assertEqual(cfg.inf_evaluators, [])

    def test_configured_gpu_launch(self):
        path = ROOT / 'configs/experiments/stage1_pixmo_cap_100k_qwen25_7b_lora.yaml'
        with patch.dict(os.environ), patch('launch_scripts.trajvlm_experiment.subprocess.run') as run:
            os.environ.pop('RANK', None)
            os.environ.pop('WORLD_SIZE', None)
            os.environ['PIXMO_CAP_DATASET'] = '/wrong/inherited/path'
            run.return_value.returncode = 0
            override = f'run.output_dir={self.root}/launch-test'
            self.assertEqual(main(['--config', str(path), '--launch', '--set', override,
                                   '--set', 'launch.gpus=[2,3]']), 0)
            command = run.call_args.args[0]
            self.assertIn('--nproc_per_node=2', command)
            self.assertNotIn('--launch', command)
            self.assertIn(override, command)
            self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '2,3')
            self.assertEqual(run.call_args.kwargs['env']['PIXMO_CAP_DATASET'],
                             load_experiment(path).recipe.data.pixmo_cap_path)
            self.assertFalse((self.root / 'launch-test').exists())

    def test_invalid_gpu_selection(self):
        for selection in ('[]', '[0,0]', '[-1]'):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.experiment(overrides=[f'launch.gpus={selection}'])

    def test_stage2_video_subset_and_metadata_only(self):
        exp = self.experiment(2, ['data.datasets=[tgif,nextqa_mc]'])
        cfg = build_experiment(exp)
        self.assertIsInstance(cfg.model, TrajVlmConfig)
        group, = cfg.data.kwargs_mixture
        self.assertEqual([d.dataset_name for d in group.datasets], ['tgif', 'nextqa_mc'])
        self.assertEqual(cfg.model.mm_preprocessor.video.num_frames, 128)
        self.assertEqual(cfg.model.mm_preprocessor.video.frames_per_clip, 16)
        self.assertEqual(cfg.model.mm_preprocessor.video.max_frames, 128)
        self.assertEqual(cfg.evaluators, [])
        self.assertEqual(cfg.inf_evaluators, [])
        self.assertIsNone(cfg.wandb)
        self.assertEqual(cfg.optimizer.connector_learning_rate, 5e-6)

    def test_complete_video_mixture_matches_authoritative_names(self):
        from launch_scripts.sft import VIDEO_ACADEMIC_DATASETS
        cfg = build_experiment(self.experiment(2, ['data.datasets=null']))
        self.assertEqual([d.dataset_name for d in cfg.data.kwargs_mixture[0].datasets],
                         [d[0] if isinstance(d, tuple) else d for d in VIDEO_ACADEMIC_DATASETS])

    def test_missing_checkpoint_is_actionable(self):
        exp = self.experiment(2, [f'checkpoint={self.root}/absent'])
        with self.assertRaisesRegex(ValueError, 'checkpoint metadata.*config.yaml'):
            build_experiment(exp)

    def test_unknown_dataset(self):
        with self.assertRaisesRegex(ValueError, 'Unknown video_academic'):
            build_experiment(self.experiment(2, ['data.datasets=[does_not_exist]']))

    def test_precedence_and_epochs_recompute(self):
        exp = self.experiment(overrides=['training.max_steps=null', 'training.epochs=4',
                                        'data.train_examples=1000', 'training.global_batch_size=100',
                                        'training.global_batch_size=200'])
        cfg = build_experiment(exp)
        self.assertEqual(cfg.global_train_batch_size, 200)
        self.assertEqual(cfg.max_duration, 20)
        self.assertEqual(cfg.stop_at, 20)

    def test_inheritance_paths_lists_and_interpolation(self):
        base = self.root / 'bases'
        base.mkdir()
        (base / 'first.yaml').write_text('run:\n  output_dir: outputs/${run.name}\nmodel:\n  segmenter_path: model.pth\nevaluation:\n  tasks: [one, two]\n')
        (base / 'second.yaml').write_text('training:\n  max_steps: 10\n')
        child = self.root / 'recipe.yaml'
        child.write_text('extends: [bases/first.yaml, bases/second.yaml]\nstage: pretrain\nrun:\n  name: inherited\ndata:\n  mixture: pixmo_cap\nevaluation:\n  enabled: true\n  tasks: [chart_qa]\n')
        exp = load_experiment(child, ['training.max_steps=30'])
        self.assertEqual(exp.recipe.run.output_dir, str(base / 'outputs/inherited'))
        self.assertEqual(exp.recipe.model.segmenter_path, str(base / 'model.pth'))
        self.assertEqual(exp.recipe.evaluation.tasks, ['chart_qa'])
        self.assertEqual(exp.recipe.training.max_steps, 30)
        self.assertEqual([Path(s['path']).name for s in exp.sources], ['first.yaml', 'second.yaml', 'recipe.yaml'])
        before = exp.recipe
        with contextlib.chdir(base):
            self.assertEqual(load_experiment(child, ['training.max_steps=30']).recipe, before)
        self.assertEqual(load_experiment(child, ['run.output_dir=relative']).recipe.run.output_dir,
                         str(self.root / 'relative'))

    def test_uri_preserved(self):
        exp = self.experiment(overrides=['model.segmenter_path=s3://bucket/model.pth'])
        self.assertEqual(exp.recipe.model.segmenter_path, 's3://bucket/model.pth')

    def test_cyclic_inheritance(self):
        path = self.root / 'loop.yaml'
        path.write_text('extends: [loop.yaml]\n')
        with self.assertRaisesRegex(ValueError, 'Cyclic extends.*loop.yaml'):
            load_experiment(path)

    def test_invalid_yaml_and_root(self):
        for content in ('[unclosed', '- list\n', 'extends: base.yaml\n'):
            with self.subTest(content=content):
                file = self.root / 'bad.yaml'
                file.write_text(content)
                with self.assertRaises(Exception):
                    load_experiment(file)

    def test_schema_errors(self):
        for override in ('unknown=1', 'training.max_stpes=4', 'training.max_steps=bad',
                         'training.compile=not-a-bool', 'run.name=???'):
            with self.subTest(override=override), self.assertRaises(OmegaConfBaseException):
                self.experiment(overrides=[override])

    def test_semantic_errors(self):
        for overrides in (
            ['schema_version=2'], ['stage=unknown'], ['training.max_steps=0'],
            ['training.epochs=4'], ['training.max_steps=null'],
            ['training.max_steps=null', 'training.epochs=2'],
            ['training.global_batch_size=-2'], ['data.num_workers=-1'],
            ['training.connector_learning_rate=-1'], ['training.vit_learning_rate=.nan'],
            ['tracking.enabled=true'], ['evaluation.enabled=true'],
            ['evaluation.tasks=[chart_qa]'], ['trainer_overrides.model={}'],
            ['trainer_overrides.optimizer.connector_learning_rate=0.1'],
            ['training.context_parallel_degree=2'], ['data.mixture=molmo2'],
            ['trainer_overrides.save_interval=-1'], ['model.num_traj=0'],
            ['trainer_overrides.max_grad_norm=.nan'], ['model.num_frames=8'],
            ['trainer_overrides.console_log_interval=0'], ['training.fsdp_precision=bf16'],
            ['trainer_overrides.microbatch_log_interval=0'], ['data.pad=longest'],
        ):
            with self.subTest(overrides=overrides), self.assertRaises((ValueError, OmegaConfBaseException)):
                build_experiment(self.experiment(overrides=overrides))

    def test_environment_resolution_is_after_overrides(self):
        with patch.dict(os.environ):
            os.environ.pop('MOLMO_DATA_DIR', None)
            with self.assertRaises(OmegaConfBaseException):
                self.experiment()
            self.assertIsNone(self.experiment(overrides=['model.segmenter_path=null']).recipe.model.segmenter_path)
        with patch.dict(os.environ, MOLMO_DATA_DIR='relative'):
            with self.assertRaisesRegex(ValueError, 'absolute'):
                self.experiment()

    def test_topology_validation(self):
        exp = self.experiment()
        cfg = build_experiment(exp)
        validate_train_config(cfg, exp.recipe, 8)
        with self.assertRaisesRegex(ValueError, 'divisible'):
            validate_train_config(cfg, exp.recipe, 3)
        cfg.model.mm_preprocessor.image.num_traj += 1
        with self.assertRaisesRegex(ValueError, 'trajectory counts'):
            validate_train_config(cfg, exp.recipe)

    def test_print_and_export_do_not_execute_training(self):
        import torch
        from olmo.data.pixmo_datasets import PixMoCap
        output = self.root / 'preview.yaml'
        def forbidden(*args, **kwargs):
            self.fail('configuration-only mode performed runtime work')
        for stage in (1, 2):
            with self.subTest(stage=stage), contextlib.ExitStack() as stack:
                for target in ('huggingface_hub.snapshot_download', 'huggingface_hub.hf_hub_download',
                               'datasets.load_from_disk', 'torch.hub.load', 'torch.cuda.set_device', 'torch.cuda.init', 'torch.cuda._lazy_init', 'torch.distributed.init_process_group',
                               'torch.load', 'wandb.init', 'datasets.load_dataset',
                               'olmo.util.prepare_torchrun_environment',
                               'olmo.train.run_trainer.run_trainer'):
                    stack.enter_context(patch(target, side_effect=forbidden))
                stack.enter_context(patch.object(PixMoCap, '__init__', side_effect=forbidden))
                stack.enter_context(patch.object(TrajVlmConfig, 'build_model', side_effect=forbidden))
                stream = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                argv = ['--config', str(STAGE1 if stage == 1 else STAGE2),
                        '--set', f'run.output_dir={self.root}/never-created',
                        '--print-config', '--export-config', str(output)]
                if stage == 2:
                    argv += ['--set', f'checkpoint={FIXTURES}/pretrain_checkpoint']
                with patch.dict(os.environ):
                    os.environ.pop('WORLD_SIZE', None)
                    self.assertEqual(main(argv), 0)
                self.assertIn('world size unknown', stream.getvalue())
                self.assertFalse((self.root / 'never-created').exists())
                restored = TrainConfig.load(output, key='train_config')
                self.assertIsInstance(restored.model, TrajVlmConfig)
                self.assertEqual(OmegaConf.load(output).validation.static, 'passed')

    def test_enabled_evaluation_and_tracking_are_explicit(self):
        exp = self.experiment(overrides=['evaluation.enabled=true', 'evaluation.tasks=[chart_qa]',
                                        'tracking.enabled=true', 'tracking.project=project', 'tracking.entity=entity'])
        cfg = build_experiment(exp)
        self.assertEqual(len(cfg.inf_evaluators), 1)
        self.assertEqual(cfg.evaluators, [])
        self.assertEqual(cfg.inf_eval_interval, 1000)
        self.assertEqual(cfg.wandb.project, 'project')

    def test_zero_workers_and_trainer_override(self):
        cfg = build_experiment(self.experiment(overrides=['data.num_workers=0', 'trainer_overrides.save_interval=50']))
        self.assertIsNone(cfg.data.prefetch_factor)
        self.assertFalse(cfg.data.persistent_workers)
        self.assertEqual(cfg.save_interval, 50)
        cfg = build_experiment(self.experiment(overrides=['trainer_overrides.microbatch_log_interval=16']))
        self.assertEqual(cfg.microbatch_log_interval, 16)

    def test_fsdp_precision_and_micro_batch(self):
        cfg = build_experiment(self.experiment())
        self.assertEqual(cfg.fsdp.precision, 'float')
        cfg = build_experiment(self.experiment(overrides=['training.fsdp_precision=mixed',
                                                          'training.micro_batch_size=4']))
        self.assertEqual(cfg.fsdp.precision, 'mixed')
        self.assertEqual(cfg.device_train_microbatch_size, 4)

    def test_training_pad_mode(self):
        cfg = build_experiment(self.experiment())
        self.assertEqual(cfg.data.pad, 'to_max')
        cfg = build_experiment(self.experiment(overrides=['data.pad=to_longest']))
        self.assertEqual(cfg.data.pad, 'to_longest')
        self.assertTrue(all(ev.data.pad == 'to_max' for ev in cfg.evaluators))

    def test_manifest_collision_and_resume_provenance(self):
        exp = self.experiment()
        cfg = build_experiment(exp)
        check_output(exp)
        with patch.dict(os.environ, UNRELATED_SECRET='must-not-be-recorded'):
            directory = write_run_record(exp, cfg, command=['experiment', '--config', 'recipe.yaml'])
        first = (directory / 'manifest.json').read_bytes()
        manifest = json.loads(first)
        self.assertEqual(len(manifest['uv_lock_sha256']), 64)
        self.assertEqual(manifest['sources'][0]['path'], exp.sources[0]['path'])
        self.assertNotIn('must-not-be-recorded', first.decode())
        for item in manifest['sources']:
            self.assertTrue((directory / item['snapshot']).is_file())
        with self.assertRaisesRegex(ValueError, 'Output already exists'):
            check_output(exp)
        with self.assertRaises(FileExistsError):
            write_run_record(exp, cfg)
        cfg.save(Path(cfg.save_folder) / 'config.yaml')
        (Path(cfg.save_folder) / 'step10').mkdir()
        exp.recipe.run.resume = True
        check_output(exp)
        cfg.optimizer.connector_learning_rate = 1e-5
        resumed = write_run_record(exp, cfg)
        self.assertIn('resumes', resumed.parts)
        self.assertEqual((directory / 'manifest.json').read_bytes(), first)
        restored = TrainConfig.load(resumed / 'train_config.yaml')
        self.assertEqual(restored.optimizer.connector_learning_rate, 1e-5)

    def test_training_dispatch_and_recording_failure(self):
        argv = ['--config', str(STAGE1), '--set', f'run.output_dir={self.root}/runtime']
        with contextlib.ExitStack() as stack:
            set_device = stack.enter_context(patch('torch.cuda.set_device'))
            prepare = stack.enter_context(patch('olmo.util.prepare_torchrun_environment'))
            stack.enter_context(patch('torch.distributed.get_world_size', return_value=8))
            stack.enter_context(patch('torch.distributed.get_rank', return_value=0))
            stack.enter_context(patch('torch.distributed.broadcast_object_list'))
            trainer = stack.enter_context(patch('olmo.train.run_trainer.run_trainer'))
            self.assertEqual(main(argv), 0)
            set_device.assert_called_once()
            prepare.assert_called_once()
            trainer.assert_called_once()
            self.assertTrue((self.root / 'runtime/experiment/manifest.json').is_file())
        argv = ['--config', str(STAGE1), '--set', f'run.output_dir={self.root}/record-error']
        with contextlib.ExitStack() as stack:
            for target in ('torch.cuda.set_device', 'olmo.util.prepare_torchrun_environment',
                           'torch.distributed.broadcast_object_list'):
                stack.enter_context(patch(target))
            stack.enter_context(patch('torch.distributed.get_world_size', return_value=8))
            stack.enter_context(patch('torch.distributed.get_rank', return_value=0))
            stack.enter_context(patch('olmo.experiment_records.write_run_record', side_effect=OSError('disk failure')))
            trainer = stack.enter_context(patch('olmo.train.run_trainer.run_trainer'))
            stream = stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            with self.assertRaises(SystemExit):
                main(argv)
            self.assertIn('disk failure', stream.getvalue())
            trainer.assert_not_called()

    def test_resume_requires_prior_run(self):
        exp = self.experiment(overrides=['run.resume=true'])
        with self.assertRaisesRegex(ValueError, 'existing config.yaml'):
            check_output(exp)

    def test_legacy_golden_configs(self):
        for stage, module, arguments in (
            ('pretrain', pretrain, ['--save_folder=/tmp/legacy-pretrain', '--max_duration=10',
                                    '--pointing=0', '--nlp=0', '--pretrained_segmenter_path=none', '--no_compile']),
            ('sft', sft, [str(FIXTURES / 'pretrain_checkpoint'), 'molmo2', '--no_compile']),
        ):
            args = module.make_parser().parse_args(arguments)
            if stage == 'pretrain':
                cfg = module.build_train_config(args, ['--run_name=legacy_pretrain'])
            else:
                cfg = module.build_train_config(args, ['--save_folder=/tmp/legacy-sft'], checkpoint_selector=lambda p:p)
            golden = json.loads((FIXTURES / f'legacy_{stage}.json').read_text())
            actual = serial(cfg)
            self.assertEqual({key: actual[key] for key in golden}, golden)
        self.assertEqual(sft.make_parser().parse_args(['checkpoint', 'molmo2']).num_frames, 128)
        self.assertEqual(sft.make_parser().parse_args(['checkpoint', 'molmo2']).frames_per_clip, 16)

    def test_recipe_and_legacy_equivalence(self):
        exp = self.experiment(overrides=['model.segmenter_path=null', 'training.max_steps=10',
                                        'data.num_workers=2', 'training.micro_batch_size=4'])
        cfg = build_experiment(exp)
        args = pretrain.make_parser().parse_args(['--save_folder', cfg.save_folder, '--max_duration=10',
                                                 '--pointing=0', '--nlp=0', '--no_compile',
                                                 '--pretrained_segmenter_path=none'])
        legacy = pretrain.build_train_config(args)
        for key in ('model', 'optimizer', 'scheduler', 'global_train_batch_size', 'max_duration', 'stop_at'):
            actual = serial(cfg)[key]
            if key == 'model':
                actual['llm'].pop('lora', None)
            self.assertEqual(actual, serial(legacy)[key], key)
        # The legacy entrypoint hardcodes 300k steps at global batch 128 and a 16k sequence.
        exp = self.experiment(2, ['training.max_steps=300000', 'training.global_batch_size=128',
                                  'training.seq_len=16384'])
        cfg = build_experiment(exp)
        args = sft.make_parser().parse_args([str(FIXTURES / 'pretrain_checkpoint'), 'molmo2',
                                             '--num_frames=128', '--frames_per_clip=16',
                                             '--device_batch_size=1', '--no_compile'])
        legacy = sft.build_train_config(args, ['--save_folder=/tmp/legacy-sft'], checkpoint_selector=lambda p:p)
        for key in ('model', 'optimizer', 'scheduler', 'max_duration', 'stop_at', 'global_train_batch_size'):
            actual = serial(cfg)[key]
            if key == 'model':
                actual['llm'].pop('lora', None)
            self.assertEqual(actual, serial(legacy)[key], key)

    def test_registry_import_orders(self):
        for order in ('pretrain,sft', 'sft,pretrain'):
            script = '\n'.join(f'import launch_scripts.trajvlm_{name}' for name in order.split(','))
            script += '\nfrom olmo.models.trajvlm_config import TrajVlmConfig\nfrom olmo.train import trainer_config\nassert trainer_config.get_model_types()["video_olmo"] is TrajVlmConfig\n'
            subprocess.run([sys.executable, '-c', script], env={**os.environ, 'PYTHONPATH': str(ROOT)},
                           check=True, capture_output=True, text=True)

    def test_shell_argument_forwarding(self):
        fakebin = self.root / 'bin'
        fakebin.mkdir()
        command = fakebin / 'torchrun'
        command.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
        command.chmod(0o755)
        env = {**os.environ, 'PATH': str(fakebin) + ':' + os.environ['PATH']}
        for stage in ('pretrain', 'sft'):
            args = ['bash', str(ROOT / f'scripts/{stage}.sh'), '--ngpus', '2', '--save_folder', str(self.root / stage)]
            if stage == 'sft':
                args += ['--pretrain_ckpt', '/tmp/ckpt']
            args += ['--optimizer.connector_learning_rate=0.0001']
            result = subprocess.run(args, env=env, check=True, capture_output=True, text=True).stdout
            self.assertIn('--nproc-per-node=2', result)
            self.assertIn('--optimizer.connector_learning_rate=0.0001', result)
            if stage == 'sft':
                self.assertIn('--num_frames\n64\n--frames_per_clip\n8\n', result)


if __name__ == '__main__':
    unittest.main()
