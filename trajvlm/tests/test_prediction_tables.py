from datetime import timedelta
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORLD_SIZE = 2
# Three examples split across ranks: unevenly, all on one rank (as the eval loader did in the
# Stage1 run), and examples without a thumbnail.
SPLITS = {"uneven": [[0, 1], [2]], "one_rank_empty": [[], [0, 1, 2]]}


class FakeTokenizer:
    eos_token_id = 1
    adds_space = False

    def decode(self, tokens):
        return " ".join(f"t{int(t)}" for t in tokens)


def _run_rank(rank, split, group_file, out_dir, with_images=True):
    import wandb
    from olmo.eval.evaluators import SavePredictions
    from olmo.eval.inf_evaluator import InfEvaluator

    torch.distributed.init_process_group('gloo', init_method=Path(group_file).as_uri(),
                                         rank=rank, world_size=WORLD_SIZE, timeout=timedelta(seconds=60))
    try:
        ids = SPLITS[split][rank]
        metadatas = [dict(image=np.full((8, 12, 3), 10 * ix, dtype=np.uint8), answer=f"caption {ix}")
                     if with_images else dict(answer=f"caption {ix}") for ix in ids]
        predictions = dict(predictions=np.array([[100 + ix, 1] for ix in ids]).reshape(len(ids), 2),
                           prompts=np.array([[200 + ix] for ix in ids]).reshape(len(ids), 1))
        evaluator = InfEvaluator([SavePredictions(out_dir, save_tokens=False, log_examples=0)])
        metrics = evaluator(predictions, metadatas, FakeTokenizer(), torch.device('cpu'), step=500)
        # Both ranks must reach this point; a mismatch in gathers would hang one of them.
        torch.distributed.barrier()
        if rank == 0:
            images = metrics.get("prediction_images", [])
            summary = dict(
                images=[isinstance(image, wandb.Image) for image in images],
                captions=[image._caption for image in images],
                html_images=metrics["prediction_table"].html.count("data:image/jpeg"),
            )
            Path(out_dir, "summary.json").write_text(json.dumps(summary))
    finally:
        torch.distributed.destroy_process_group()


class PredictionTableTests(unittest.TestCase):
    def run_ranks(self, split, with_images=True):
        with tempfile.TemporaryDirectory() as tmp:
            # A gather mismatch hangs; the default Gloo timeout turns that into a failure.
            mp.spawn(_run_rank, args=(split, str(Path(tmp, "group")), tmp, with_images), nprocs=WORLD_SIZE)
            return json.loads(Path(tmp, "summary.json").read_text())

    def check_every_example(self, summary):
        self.assertEqual(summary["images"], [True, True, True])
        self.assertEqual(summary["captions"], [
            f"PREDICTION:\nt{100 + ix} t1\n\nREFERENCE:\ncaption {ix}\n\nPROMPT:\nt{200 + ix}" for ix in range(3)])
        self.assertEqual(summary["html_images"], 3)

    def test_every_example_is_logged_with_its_image(self):
        self.check_every_example(self.run_ranks("uneven"))

    def test_rank_without_examples_still_joins_every_gather(self):
        self.check_every_example(self.run_ranks("one_rank_empty"))

    def test_examples_without_images(self):
        summary = self.run_ranks("uneven", with_images=False)
        self.assertEqual(summary["images"], [])
        self.assertEqual(summary["html_images"], 0)


if __name__ == '__main__':
    unittest.main()
