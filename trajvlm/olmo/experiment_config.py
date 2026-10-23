"""Typed YAML recipes for TrajVLM. Configuration construction never starts training."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from omegaconf import MISSING, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from olmo.nn.peft import PeftConfig


@dataclass
class Run:
    name: str = MISSING
    output_dir: str = MISSING
    resume: bool = False
    seed: int = 6198


@dataclass
class Model:
    llm: Optional[str] = None
    segmenter_path: Optional[str] = None
    num_traj: Optional[int] = None
    image_res: Optional[int] = None
    # Hide trajectories assigned no patch from the LLM; sft inherits it from the checkpoint.
    mask_empty_trajectories: Optional[bool] = None
    num_frames: Optional[int] = None
    frames_per_clip: Optional[int] = None


@dataclass
class Data:
    mixture: str = MISSING
    pixmo_cap_path: Optional[str] = None
    datasets: Optional[List[str]] = None
    # Expected examples drawn from each dataset over the run (sampling rates proportional
    # to these counts); replaces data.datasets and may name any registered video dataset.
    dataset_examples: Optional[Dict[str, int]] = None
    # Benchmarks whose videos are dropped from every training dataset (eval_overlap.EVAL_VIDEO_IDS).
    exclude_eval_videos: List[str] = field(default_factory=list)
    train_examples: Optional[int] = None
    num_workers: int = 2
    prefetch_factor: int = 4
    # Training text padding: to_max pads every example to training.seq_len; to_longest pads
    # to the longest example in each micro-batch (same truncation, less wasted compute).
    pad: str = 'to_max'


@dataclass
class Training:
    max_steps: Optional[int] = None
    epochs: Optional[int] = None
    global_batch_size: int = 128
    micro_batch_size: int = 1
    seq_len: int = 2536
    connector_learning_rate: float = 2e-4
    vit_learning_rate: float = 6e-6
    llm_learning_rate: float = 2e-5
    connector_warmup: int = 200
    vit_warmup: int = 2000
    llm_warmup: int = 2000
    compile: bool = False
    context_parallel_degree: int = 1
    # FSDP parameter/reduction dtypes: float, mixed (bf16 params, fp32 reduce) or
    # pure (bf16 both). None keeps the stage default.
    fsdp_precision: Optional[str] = None


@dataclass
class Evaluation:
    enabled: bool = False
    tasks: List[str] = field(default_factory=list)
    loss_tasks: List[str] = field(default_factory=list)
    interval: int = 1000
    max_examples: int = 2048


@dataclass
class Tracking:
    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None


@dataclass
class Launch:
    # Explicit device IDs; --launch uses one process per listed GPU on one node.
    gpus: List[int] = field(default_factory=lambda: [0])


@dataclass
class Recipe:
    schema_version: int = 1
    stage: str = MISSING
    run: Run = field(default_factory=Run)
    model: Model = field(default_factory=Model)
    data: Data = field(default_factory=Data)
    training: Training = field(default_factory=Training)
    launch: Launch = field(default_factory=Launch)
    evaluation: Evaluation = field(default_factory=Evaluation)
    tracking: Tracking = field(default_factory=Tracking)
    peft: Optional[PeftConfig] = None
    checkpoint: Optional[str] = None
    trainer_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    recipe: Recipe
    sources: List[Dict[str, str]]
    overrides: List[str]


PATH_KEYS = {"run.output_dir", "model.segmenter_path", "checkpoint", "data.pixmo_cap_path"}
# Structural settings must use public recipe fields so all derived values agree.
TRAINER_OVERRIDES = {
    "save_interval", "save_num_checkpoints_to_keep", "max_grad_norm",
    "console_log_interval", "softmax_auxiliary_loss", "softmax_auxiliary_loss_scale",
    "microbatch_log_interval",
}


def _declared(conf, key):
    value = OmegaConf.to_container(conf, resolve=False)
    for part in key.split('.'):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


def load_experiment(path, overrides=()) -> Experiment:
    """Merge first, then resolve interpolation and defining-file-relative paths."""
    path = Path(path).resolve()
    sources = []

    def visit(current, stack):
        if current in stack:
            raise ValueError('Cyclic extends: ' + ' -> '.join(map(str, [*stack, current])))
        text = current.read_text()
        raw = OmegaConf.create(text)
        if not OmegaConf.is_dict(raw):
            raise ValueError(f"{current}: recipe must be a YAML mapping")
        bases = raw.pop('extends', [])
        if not OmegaConf.is_list(bases) and not isinstance(bases, list):
            raise ValueError(f"{current}: extends must be a list")
        merged, origins = OmegaConf.create({}), {}
        for base in bases:
            if not isinstance(base, str):
                raise ValueError(f"{current}: extends entries must be paths")
            inherited, inherited_origins = visit((current.parent / base).resolve(), [*stack, current])
            merged = OmegaConf.merge(merged, inherited)
            origins.update(inherited_origins)
        origins.update({key: current.parent for key in PATH_KEYS if _declared(raw, key)})
        sources.append({'path': str(current), 'content': text})
        return OmegaConf.merge(merged, raw), origins

    conf, origins = visit(path, [])
    for item in overrides:
        if '=' not in item or item.startswith('--'):
            raise ValueError('--set expects key=value')
        update = OmegaConf.from_dotlist([item])
        if OmegaConf.missing_keys(update):
            raise MissingMandatoryValue(f"CLI override is missing a value: {item}")
        origins.update({key: path.parent for key in PATH_KEYS if _declared(update, key)})
        conf = OmegaConf.merge(conf, update)
    typed = OmegaConf.merge(OmegaConf.structured(Recipe), conf)
    # Keep raw expressions for the environment-root check. Resolve only after
    # CLI overrides, so a child may supply values referenced by a base file.
    raw_paths = {key: _raw_value(typed, key) for key in PATH_KEYS}
    OmegaConf.resolve(typed)
    for key in PATH_KEYS:
        value = OmegaConf.select(typed, key)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key}: expected a nonempty path")
        if urlparse(value).scheme:
            continue
        local = Path(value).expanduser()
        if '${oc.env:' in str(raw_paths[key]) and not local.is_absolute():
            raise ValueError(f"{key}: environment-based paths must be absolute")
        OmegaConf.update(typed, key, str((origins.get(key, path.parent) / local).resolve()), merge=False)
    recipe = OmegaConf.to_object(typed)
    validate_recipe(recipe)
    return Experiment(recipe, sources, list(overrides))


def _raw_value(conf, key):
    value = OmegaConf.to_container(conf, resolve=False)
    for part in key.split('.'):
        value = value[part]
    return value


def validate_recipe(r):
    if (not r.launch.gpus or len(set(r.launch.gpus)) != len(r.launch.gpus)
            or any(gpu < 0 for gpu in r.launch.gpus)):
        raise ValueError('launch.gpus must contain unique nonnegative GPU IDs')
    if r.schema_version != 1:
        raise ValueError('schema_version must be 1')
    if r.stage not in ('pretrain', 'sft'):
        raise ValueError('stage must be pretrain or sft')
    if not r.run.name.strip() or not r.run.output_dir.strip():
        raise ValueError('run.name and run.output_dir must be nonempty')
    if urlparse(r.run.output_dir).scheme:
        raise ValueError('run.output_dir currently requires a local filesystem path')
    t = r.training
    if r.peft is not None:
        r.peft.validate()
        if r.stage == 'sft' and r.peft.mode != 'connector_lora':
            raise ValueError('sft requires peft.mode=connector_lora')
    if (t.max_steps is None) == (t.epochs is None):
        raise ValueError('Specify exactly one of training.max_steps or training.epochs')
    positive = {'global_batch_size': t.global_batch_size, 'micro_batch_size': t.micro_batch_size,
                'seq_len': t.seq_len, 'context_parallel_degree': t.context_parallel_degree,
                'prefetch_factor': r.data.prefetch_factor}
    for name, value in {'num_frames': r.model.num_frames, 'frames_per_clip': r.model.frames_per_clip,
                        'max_steps': t.max_steps, 'epochs': t.epochs,
                        'train_examples': r.data.train_examples, 'num_traj': r.model.num_traj,
                        'image_res': r.model.image_res}.items():
        if value is not None:
            positive[name] = value
    if any(v <= 0 for v in positive.values()):
        raise ValueError(f'Values must be positive: {positive}')
    if r.data.num_workers < 0:
        raise ValueError('data.num_workers must be nonnegative')
    if r.data.pad not in ('to_max', 'to_longest'):
        raise ValueError('data.pad must be to_max or to_longest')
    if t.fsdp_precision not in (None, 'float', 'mixed', 'pure'):
        raise ValueError('training.fsdp_precision must be float, mixed or pure')
    if t.epochs is not None and r.data.train_examples is None:
        raise ValueError('training.epochs requires data.train_examples')
    for prefix in ('connector', 'vit', 'llm'):
        lr = getattr(t, prefix + '_learning_rate')
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError(f'training.{prefix}_learning_rate must be positive and finite')
        if getattr(t, prefix + '_warmup') < 0:
            raise ValueError(f'training.{prefix}_warmup must be nonnegative')
    if r.tracking.enabled and (not r.tracking.project or not r.tracking.entity):
        raise ValueError('tracking.enabled requires tracking.project and tracking.entity')
    if r.evaluation.enabled and (r.evaluation.interval <= 0 or r.evaluation.max_examples <= 0 or
                                 not (r.evaluation.tasks or r.evaluation.loss_tasks)):
        raise ValueError('enabled evaluation requires targets and positive interval/max_examples')
    if not r.evaluation.enabled and (r.evaluation.tasks or r.evaluation.loss_tasks):
        raise ValueError('evaluation targets require evaluation.enabled=true')
    forbidden = set(r.trainer_overrides) - TRAINER_OVERRIDES
    if forbidden:
        raise ValueError(f'trainer_overrides disallows {sorted(forbidden)}; use public recipe fields')
    if r.stage == 'pretrain':
        if r.model.num_frames is not None or r.model.frames_per_clip is not None:
            raise ValueError('pretrain is image-only; omit model.num_frames and model.frames_per_clip')
        if (r.checkpoint is not None or r.data.mixture != 'pixmo_cap' or r.data.datasets is not None
                or r.data.dataset_examples is not None or r.data.exclude_eval_videos):
            raise ValueError('pretrain requires data.mixture=pixmo_cap, no subset and no checkpoint')
        if t.context_parallel_degree != 1:
            raise ValueError('pretrain currently requires context_parallel_degree=1')
    else:
        if not r.checkpoint:
            raise ValueError('sft requires checkpoint pointing to a checkpoint directory with config.yaml')
        if r.model.num_frames is None or r.model.frames_per_clip is None:
            raise ValueError('sft requires model.num_frames and model.frames_per_clip')
        if urlparse(r.checkpoint).scheme:
            raise ValueError('sft requires local checkpoint metadata (a mounted NAS path is supported)')
        if r.model.llm is not None or r.model.segmenter_path is not None:
            raise ValueError('sft inherits llm and segmenter from checkpoint; omit those model fields')
        if r.data.mixture != 'video_academic':
            raise ValueError('sft recipe supports data.mixture=video_academic; legacy CLI supports molmo2')
        if r.data.datasets is not None and (not r.data.datasets or len(set(r.data.datasets)) != len(r.data.datasets)):
            raise ValueError('data.datasets must be a nonempty list of unique video dataset names')
        if r.data.dataset_examples is not None:
            if r.data.datasets is not None:
                raise ValueError('set only one of data.datasets and data.dataset_examples')
            if not r.data.dataset_examples or any(n <= 0 for n in r.data.dataset_examples.values()):
                raise ValueError('data.dataset_examples must map dataset names to positive counts')
        if len(set(r.data.exclude_eval_videos)) != len(r.data.exclude_eval_videos):
            raise ValueError('data.exclude_eval_videos must be unique benchmark names')


def build_experiment(experiment):
    """Generate final TrainConfig after composing every public input."""
    from launch_scripts import trajvlm_pretrain as pretrain, trajvlm_sft as sft
    from olmo.data.data_loader import KwargsMixture, WeightedDataset
    from olmo.eval.eval_utils import get_evaluation
    from olmo.train.trainer_config import FSDPPrecision, WandbConfig

    r = experiment.recipe
    t = r.training
    duration = t.max_steps if t.max_steps is not None else t.epochs * math.ceil(r.data.train_examples / t.global_batch_size)

    def configure(cfg):
        cfg.peft = r.peft
        if r.peft is not None:
            cfg.ft_connector, cfg.ft_vit = True, False
            cfg.ft_llm = r.peft.mode == 'connector_lora'
            cfg.model.llm.lora = r.peft.lora if cfg.ft_llm else None
        cfg.run_name, cfg.save_folder, cfg.seed = r.run.name, r.run.output_dir, r.run.seed
        cfg.allow_resume, cfg.save_overwrite = r.run.resume, False
        cfg.max_duration = cfg.stop_at = duration
        cfg.global_train_batch_size = t.global_batch_size
        cfg.device_train_microbatch_size = t.micro_batch_size
        cfg.compile_loss = t.compile
        if t.fsdp_precision is not None:
            cfg.fsdp.precision = FSDPPrecision(t.fsdp_precision)
        cfg.data.pad = r.data.pad
        cfg.data.num_workers = r.data.num_workers
        cfg.data.persistent_workers = r.data.num_workers > 0
        cfg.data.prefetch_factor = r.data.prefetch_factor if r.data.num_workers > 0 else None
        for prefix in ('connector', 'vit', 'llm'):
            setattr(cfg.optimizer, prefix + '_learning_rate', getattr(t, prefix + '_learning_rate'))
            setattr(cfg.scheduler, prefix + '_t_warmup', getattr(t, prefix + '_warmup'))
        cfg.wandb = (WandbConfig(name=r.run.name, project=r.tracking.project, entity=r.tracking.entity)
                     if r.tracking.enabled else None)
        cfg.evaluators, cfg.inf_evaluators = [], []
        cfg.eval_interval = cfg.inf_eval_interval = -1
        if r.evaluation.enabled:
            for inference, tasks in ((True, r.evaluation.tasks), (False, r.evaluation.loss_tasks)):
                for task in tasks:
                    ev = get_evaluation(task, None if inference else t.seq_len,
                                        for_inference=inference, device_batch_size=t.micro_batch_size,
                                        max_examples=r.evaluation.max_examples, num_workers=r.data.num_workers)
                    ev.data.seed = r.run.seed
                    ev.data.persistent_workers = r.data.num_workers > 0
                    ev.data.prefetch_factor = r.data.prefetch_factor if r.data.num_workers > 0 else None
                    if inference:
                        # As in the SFT launchers: generation batches are unpadded and only the
                        # question text needs to fit.
                        ev.data.pad = None
                        ev.data.max_text_seq_len = 128
                    (cfg.inf_evaluators if inference else cfg.evaluators).append(ev)
            cfg.eval_interval = r.evaluation.interval if cfg.evaluators else -1
            cfg.inf_eval_interval = r.evaluation.interval if cfg.inf_evaluators else -1

    if r.stage == 'pretrain':
        args = pretrain.make_parser().parse_args([
            '--save_folder', r.run.output_dir, '--llm', r.model.llm or 'qwen3_4b_instruct',
            '--pretrained_segmenter_path', r.model.segmenter_path or 'none',
            '--num_traj', str(r.model.num_traj or 128), '--image_res', str(r.model.image_res or 378),
            '--max_duration', str(duration), '--global_batch_size', str(t.global_batch_size),
            '--device_train_microbatch_size', str(t.micro_batch_size), '--seq_len', str(t.seq_len),
            '--pointing', '0', '--nlp', '0', *([] if t.compile else ['--no_compile']),
            *(['--mask_empty_trajectories'] if r.model.mask_empty_trajectories else []),
        ])
        cfg = pretrain.build_train_config(args, configure=configure)
    else:
        from launch_scripts.sft import VIDEO_ACADEMIC_DATASETS
        from olmo.data.eval_overlap import EVAL_VIDEO_IDS
        from olmo.data.get_dataset import get_dataset_class_by_name
        datasets = [WeightedDataset.build(x) for x in VIDEO_ACADEMIC_DATASETS]
        if r.data.datasets is not None:
            by_name = {x.dataset_name: x for x in datasets}
            unknown = set(r.data.datasets) - by_name.keys()
            if unknown:
                raise ValueError(f'Unknown video_academic datasets: {sorted(unknown)}')
            datasets = [by_name[name] for name in r.data.datasets]
        elif r.data.dataset_examples is not None:
            unknown = []
            for name in r.data.dataset_examples:
                try:
                    get_dataset_class_by_name(name)
                except NotImplementedError:
                    unknown.append(name)
            if unknown:
                raise ValueError(f'Unknown video datasets: {sorted(unknown)}')
            # root_size_factor=0 makes the rate the count itself rather than sqrt(dataset size).
            datasets = [WeightedDataset(name, sampling_rate=float(n), root_size_factor=0)
                        for name, n in r.data.dataset_examples.items()]
        unknown = set(r.data.exclude_eval_videos) - EVAL_VIDEO_IDS.keys()
        if unknown:
            raise ValueError(f'Unknown data.exclude_eval_videos benchmarks: {sorted(unknown)}')
        if r.data.exclude_eval_videos:
            for task in datasets:
                task.exclude_eval_videos = list(r.data.exclude_eval_videos)
        checkpoint_config = Path(r.checkpoint) / 'config.yaml'
        if not checkpoint_config.is_file():
            raise ValueError(f'Stage2 requires checkpoint metadata: {checkpoint_config}')

        # Dispatch through the shared TrajVLM checkpoint type registration.
        from olmo.models.trajvlm_config import TrajVlmConfig
        def model_loader(checkpoint, **kwargs):
            return sft.get_trajvlm_model(checkpoint, config_loader=TrajVlmConfig.load, **kwargs)

        args = sft.make_parser().parse_args([
            r.checkpoint, 'video_academic', '--seq_len', str(t.seq_len),
            '--device_batch_size', str(t.micro_batch_size), '--num_frames', str(r.model.num_frames),
            '--frames_per_clip', str(r.model.frames_per_clip), '--cp_degree', str(t.context_parallel_degree),
            '--num_workers', str(r.data.num_workers), *([] if t.compile else ['--no_compile']),
        ])
        cfg = sft.build_train_config(args, checkpoint_selector=lambda p: p, model_loader=model_loader,
                                    training_mixture=[KwargsMixture(1.0, datasets, name='video_academic')],
                                    evaluation_tasks=([], []), configure=configure)
    conf = OmegaConf.structured(cfg)
    conf = OmegaConf.merge(conf, r.trainer_overrides)
    OmegaConf.resolve(conf)
    cfg = OmegaConf.to_object(conf)
    validate_train_config(cfg, r)
    return cfg


def validate_train_config(cfg, recipe, world_size=None):
    vis = cfg.model.vision_backbone
    if vis.num_traj <= 0 or vis.frames_per_pool <= 0:
        raise ValueError('checkpoint trajectory count and frames_per_pool must be positive')
    for pre in (cfg.model.mm_preprocessor.image, cfg.model.mm_preprocessor.video):
        if pre.num_traj != vis.num_traj:
            raise ValueError('model.num_traj and preprocessing trajectory counts disagree')
        if pre.image_res != cfg.model.vision_backbone.vit.image_default_input_size[0]:
            raise ValueError('model.image_res must match the SigLIP2 input resolution')
    video = cfg.model.mm_preprocessor.video
    if video.frames_per_clip > 1 and vis.frames_per_pool != video.frames_per_clip:
        raise ValueError('vision backbone frames_per_pool must equal video frames_per_clip')
    if recipe.model.num_traj is not None and vis.num_traj != recipe.model.num_traj:
        raise ValueError('model.num_traj disagrees with inherited checkpoint')
    if recipe.model.image_res is not None and cfg.model.mm_preprocessor.image.image_res != recipe.model.image_res:
        raise ValueError('model.image_res disagrees with inherited checkpoint')
    if (recipe.model.mask_empty_trajectories is not None
            and vis.mask_empty_trajectories != recipe.model.mask_empty_trajectories):
        raise ValueError('model.mask_empty_trajectories disagrees with inherited checkpoint')
    if cfg.max_duration != cfg.stop_at:
        raise ValueError('max_duration and stop_at must agree')
    for group in cfg.data.kwargs_mixture or []:
        if not math.isfinite(group.rate) or group.rate <= 0:
            raise ValueError('mixture weights must be positive and finite')
    if cfg.save_interval is None or cfg.save_interval <= 0 or cfg.save_num_checkpoints_to_keep < 1:
        raise ValueError('save_interval and checkpoint retention must be positive')
    if cfg.max_grad_norm is None or not math.isfinite(cfg.max_grad_norm) or cfg.max_grad_norm <= 0:
        raise ValueError('max_grad_norm must be positive and finite')
    if cfg.console_log_interval <= 0 or not math.isfinite(cfg.softmax_auxiliary_loss_scale) or cfg.softmax_auxiliary_loss_scale < 0:
        raise ValueError('console_log_interval must be positive and auxiliary loss scale nonnegative and finite')
    if cfg.microbatch_log_interval is not None and cfg.microbatch_log_interval <= 0:
        raise ValueError('microbatch_log_interval must be positive when set')
    if world_size is not None:
        cp = recipe.training.context_parallel_degree
        if world_size <= 0 or world_size % cp:
            raise ValueError('world size must be positive and divisible by context parallel degree')
        dp = world_size // cp
        if cfg.global_train_batch_size % (dp * cfg.device_train_microbatch_size):
            raise ValueError('global batch size must be divisible by data-parallel size * micro batch size')
