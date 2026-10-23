"""Drop training examples whose video also appears in an evaluation benchmark.

Videos are matched by file stem, which is the source id for the overlaps that matter here:
FAVOR-Bench keeps Charades ids (e.g. ``00X3U.mp4``), the same ids Charades-STA and STAR use.
"""
import functools
import json
import logging
from os.path import basename, exists, join, splitext

from olmo.data.academic_video_datasets import FAVOR_HOME, MOTION_BENCH_HOME
from olmo.io import read_file

log = logging.getLogger(__name__)


def _stem(path):
    return splitext(basename(path))[0]


def _favor_bench():
    path = join(FAVOR_HOME, "question_perspective.json")
    if not exists(path):
        raise FileNotFoundError(f"{path} not found; download favor_train first")
    return {_stem(q["video_name"]) for q in json.loads(read_file(path))}


def _motionbench():
    path = join(MOTION_BENCH_HOME, "MotionBench", "video_info.meta.jsonl")
    if not exists(path):
        raise FileNotFoundError(f"{path} not found; download motionbench_train first")
    return {_stem(json.loads(line)["video_path"]) for line in read_file(path).splitlines() if line}


EVAL_VIDEO_IDS = {"favor_bench": _favor_bench, "motionbench": _motionbench}


@functools.lru_cache()
def eval_video_ids(benchmarks):
    unknown = set(benchmarks) - EVAL_VIDEO_IDS.keys()
    if unknown:
        raise ValueError(f"Unknown benchmarks for video exclusion: {sorted(unknown)}")
    return frozenset().union(*(EVAL_VIDEO_IDS[name]() for name in benchmarks))


def exclude_eval_videos(dataset, name, benchmarks):
    """Remove, in place, examples of a loaded `DatasetBase` whose video is an eval video."""
    data = getattr(dataset, "data", None)
    if not isinstance(data, list) or not all(isinstance(ex, dict) and isinstance(ex.get("video"), str)
                                             for ex in data):
        raise ValueError(f"{name}: eval-video exclusion needs loaded examples with a 'video' path")
    ids = eval_video_ids(tuple(sorted(benchmarks)))
    dataset.data = [ex for ex in data if _stem(ex["video"]) not in ids]
    removed = len(data) - len(dataset.data)
    if removed:
        log.info(f"{name}: dropped {removed} / {len(data)} examples overlapping {sorted(benchmarks)}")
    return removed
