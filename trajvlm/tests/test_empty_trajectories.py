"""CPU tests that trajectories assigned no patch are zeroed and hidden from the LLM."""
from dataclasses import replace
import os
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from launch_scripts.trajvlm_pretrain import build_model_cfg

NUM_TRAJ = 128
# Image tokens 0, 5 and 9 carry content; the rest belong to unassigned trajectories.
VALID = torch.zeros(NUM_TRAJ, dtype=torch.bool)
VALID[[0, 5, 9]] = True


def tiny_model():
    cfg = build_model_cfg("qwen2.5_7b_instruct", pretrained_segmenter_path=None,
                          mask_empty_trajectories=True)
    cfg.llm = replace(cfg.llm, init_path=None, d_model=64, n_heads=2, n_kv_heads=1, head_dim=32,
                      mlp_hidden_size=128, n_layers=2, max_sequence_length=512)
    cfg.vision_backbone.vit = replace(cfg.vision_backbone.vit, init_path=None)
    torch.manual_seed(0)
    model = cfg.build_model(device="cpu")
    model.reset_parameters()
    return model.eval()


# Building the segmenter needs the DINOv3 source tree (trajvlm/scripts/data_env.sh).
@unittest.skipUnless(os.environ.get("DINOV3_SOURCE_DIR") or os.environ.get("MOLMO_DATA_DIR"),
                     "needs DINOV3_SOURCE_DIR or MOLMO_DATA_DIR")
class EmptyTrajectories(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        tok = cls.model.config.llm.build_tokenizer()
        prompt = tok.encode("Describe the image.")
        cls.input_ids = torch.tensor([[tok.image_start_token_id] + [tok.image_patch_token_id] * NUM_TRAJ
                                      + [tok.image_end_token_id] + prompt])
        cls.batch = dict(input_ids=cls.input_ids, images=torch.zeros(1, 1, 1, 3),
                         cum_image_bounds=[torch.tensor([1])])

    def run_with_features(self, features, fn):
        """Run `fn` with the backbone replaced by one that returns `features` and `VALID`."""
        backbone = self.model.vision_backbone
        original = backbone.forward
        backbone.forward = lambda *args, **kwargs: (features * VALID[:, None], VALID)
        try:
            with torch.no_grad():
                return fn()
        finally:
            backbone.forward = original

    def features(self, seed):
        return torch.randn(NUM_TRAJ, self.model.config.llm.d_model,
                           generator=torch.Generator().manual_seed(seed))

    def with_hidden_positions_changed(self, features, fn, delta=5.0):
        """Run `fn` with only the inputs at unassigned positions changed.

        Those positions hold just the <im_patch> embedding, so shift that embedding and cancel the
        shift at the valid positions through their features.
        """
        wte = self.model.transformer.wte
        row = self.input_ids[0, 1].item() - wte.embedding.shape[0]
        original = wte.new_embedding.data.clone()
        compensated = features.clone()
        compensated[VALID] -= delta
        wte.new_embedding.data[row] += delta
        try:
            return self.run_with_features(compensated, fn)
        finally:
            wte.new_embedding.data.copy_(original)

    def last_logits(self):
        return self.model(self.input_ids, images=self.batch["images"],
                          cum_image_bounds=self.batch["cum_image_bounds"]).logits[0, -1]

    def generate(self):
        return self.model.generate(dict(self.batch), max_steps=4).token_ids

    def test_unassigned_tokens_do_not_reach_the_text(self):
        features = self.features(0)
        reference = self.run_with_features(features, self.last_logits)
        shifted = features.clone()
        shifted[VALID] += 1.0
        self.assertFalse(torch.allclose(self.run_with_features(shifted, self.last_logits), reference),
                         "valid tokens must reach the text")
        torch.testing.assert_close(self.with_hidden_positions_changed(features, self.last_logits), reference)

    def test_generation_keeps_them_hidden(self):
        features = self.features(1)
        prefill = self.run_with_features(features, lambda: self.model(
            self.input_ids, images=self.batch["images"], cum_image_bounds=self.batch["cum_image_bounds"],
            use_cache=True))
        self.assertTrue(prefill.internal["blocked_image_tokens"][0, 1:NUM_TRAJ + 1].equal(~VALID))
        reference = self.run_with_features(features, self.generate)
        self.assertTrue(self.with_hidden_positions_changed(features, self.generate).equal(reference))

    def test_backbone_zeroes_unassigned_trajectories(self):
        backbone = self.model.vision_backbone
        images = torch.rand(1, 1, 1, 378 * 378 * 3)
        with torch.no_grad():
            features, valid = backbone(images, cum_image_bounds=[torch.tensor([1])])
        self.assertEqual(features.shape, (NUM_TRAJ, self.model.config.llm.d_model))
        self.assertEqual(valid.shape, (NUM_TRAJ,))
        self.assertTrue(valid.any())
        self.assertTrue((features[~valid] == 0).all())
        self.assertTrue((features[valid].abs().sum(-1) > 0).all())

    def test_backbone_keeps_every_trajectory_by_default(self):
        backbone = self.model.vision_backbone
        backbone.config.mask_empty_trajectories = False
        try:
            with torch.no_grad():
                features = backbone(torch.rand(1, 1, 1, 378 * 378 * 3), cum_image_bounds=[torch.tensor([1])])
        finally:
            backbone.config.mask_empty_trajectories = True
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape, (NUM_TRAJ, self.model.config.llm.d_model))


if __name__ == "__main__":
    unittest.main()
