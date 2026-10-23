"""CPU tests for FAVOR-Train loading and eval-video exclusion on small local fixtures."""
import json
import tempfile
import unittest
from os.path import join
from pathlib import Path
from unittest.mock import patch

from olmo.data import academic_video_datasets as avd
from olmo.data import eval_overlap


class EvalOverlap(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        (self.home / "videos/FAVOR-Train").mkdir(parents=True)
        rows = [dict(video_name="yt_1.mp4", subset="Koala36M", caption="a man jumps"),
                dict(video_name="ABCDEEGO.mp4", subset="charadesego", caption="I open a door"),
                dict(video_name="yt_2.mp4", subset="Koala36M", caption="not downloaded")]
        (self.home / "train.json").write_text(json.dumps(rows))
        (self.home / "question_perspective.json").write_text(json.dumps(
            [dict(video_name="00X3U.mp4"), dict(video_name="00X3U.mp4"), dict(video_name="00_001823.mp4")]))
        for name in ("yt_1.mp4", "ABCDEEGO.mp4"):
            (self.home / "videos/FAVOR-Train" / name).touch()
        for target, value in ((avd, "FAVOR_HOME"), (eval_overlap, "FAVOR_HOME")):
            patcher = patch.object(target, value, str(self.home))
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(avd.FavorTrain, "VIDEO_DIR", str(self.home / "videos/FAVOR-Train"))
        patcher.start()
        self.addCleanup(patcher.stop)
        eval_overlap.eval_video_ids.cache_clear()
        self.addCleanup(eval_overlap.eval_video_ids.cache_clear)

    def test_favor_train_subsets_and_missing_videos(self):
        third_person = avd.FavorTrain("train")
        self.assertEqual([Path(ex["video"]).name for ex in third_person.data], ["yt_1.mp4"])
        self.assertEqual(third_person.data[0]["message_list"],
                         [dict(text="a man jumps", style="video_motion_caption")])
        self.assertEqual(len(avd.FavorTrain("train", subsets=None)), 2)

    def test_exclude_by_video_stem(self):
        class Loaded:
            data = [{"video": "/x/Charades_v1/00X3U.mp4"}, {"video": "/x/Charades_v1/ZZZZZ.mp4"}]
        dataset = Loaded()
        self.assertEqual(eval_overlap.exclude_eval_videos(dataset, "star", ["favor_bench"]), 1)
        self.assertEqual(dataset.data, [{"video": "/x/Charades_v1/ZZZZZ.mp4"}])

    def test_unsupported_dataset_and_benchmark(self):
        class Unloaded:
            data = None
        with self.assertRaisesRegex(ValueError, "'video' path"):
            eval_overlap.exclude_eval_videos(Unloaded(), "hf", ["favor_bench"])
        with self.assertRaisesRegex(ValueError, "Unknown benchmarks"):
            eval_overlap.eval_video_ids(("mvbench",))


if __name__ == "__main__":
    unittest.main()
