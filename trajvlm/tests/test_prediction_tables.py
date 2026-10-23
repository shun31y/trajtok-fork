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
# Three examples split unevenly across ranks, as the qualitative tasks are
EXAMPLES_PER_RANK = [[0, 1], [2]]


class FakeTokenizer:
    eos_token_id = 1
    adds_space = False

    def decode(self, tokens):
        return " ".join(f"t{int(t)}" for t in tokens)


def _run_rank(rank, group_file, out_dir):
    import wandb
    from olmo.eval.evaluators import SavePredictions
    from olmo.eval.inf_evaluator import InfEvaluator

    torch.distributed.init_process_group('gloo', init_method=Path(group_file).as_uri(),
                                         rank=rank, world_size=WORLD_SIZE)
    try:
        ids = EXAMPLES_PER_RANK[rank]
        metadatas = [dict(image=np.full((8, 12, 3), 10 * ix, dtype=np.uint8), answer=f"caption {ix}")
                     for ix in ids]
        predictions = dict(predictions=np.array([[100 + ix, 1] for ix in ids]),
                           prompts=np.array([[200 + ix] for ix in ids]))
        evaluator = InfEvaluator([SavePredictions(out_dir, save_tokens=False, log_examples=0)])
        metrics = evaluator(predictions, metadatas, FakeTokenizer(), torch.device('cpu'), step=500)
        if rank == 0:
            images = metrics["prediction_images"]
            summary = dict(
                images=[isinstance(image, wandb.Image) for image in images],
                captions=[image._caption for image in images],
                html_images=metrics["prediction_table"].html.count("data:image/jpeg"),
            )
            Path(out_dir, "summary.json").write_text(json.dumps(summary))
    finally:
        torch.distributed.destroy_process_group()


class PredictionTableTests(unittest.TestCase):
    def test_every_example_is_logged_with_its_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(_run_rank, args=(str(Path(tmp, "group")), tmp), nprocs=WORLD_SIZE)
            summary = json.loads(Path(tmp, "summary.json").read_text())
        self.assertEqual(summary["images"], [True, True, True])
        self.assertEqual(summary["captions"], [
            f"PREDICTION:\nt{100 + ix} t1\n\nREFERENCE:\ncaption {ix}\n\nPROMPT:\nt{200 + ix}" for ix in range(3)])
        self.assertEqual(summary["html_images"], 3)


if __name__ == '__main__':
    unittest.main()
