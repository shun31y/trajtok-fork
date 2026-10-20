"""CPU checks that "to_longest" padding keeps real tokens identical to "to_max"."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo import tokenizer
from olmo.preprocessing.multimodal_collator import MMCollator
from olmo.preprocessing.preprocessor_utils import TensorSpec

SPECIAL = {tokenizer.IM_END_TOKEN: 900, tokenizer.IM_START_TOKEN: 901, tokenizer.IM_COL_TOKEN: 902,
           tokenizer.IMAGE_LOW_RES_TOKEN: 903, tokenizer.IMAGE_PATCH_TOKEN: 904}
SEQ_LEN = 16
SHAPES = {'tokens': TensorSpec([SEQ_LEN], np.int64), 'images': TensorSpec([2, 3], np.float32)}


def example(n, image_rows=1):
    return dict(input_tokens=np.arange(1, n + 1), target_tokens=np.arange(2, n + 2),
                loss_masks=np.ones(n, dtype=np.float32), position_ids=np.arange(n),
                images=np.ones([image_rows, 3], dtype=np.float32))


class CollatorPadding(unittest.TestCase):
    def collate(self, pad, batch):
        return MMCollator(SPECIAL, SHAPES, include_metadata=False, pad=pad)(batch)

    def test_to_longest_matches_to_max_prefix(self):
        batch = [example(5), example(9, image_rows=2)]
        full = self.collate('to_max', [dict(x) for x in batch])
        short = self.collate('to_longest', [dict(x) for x in batch])
        for key in ('input_ids', 'labels', 'loss_masks', 'position_ids'):
            self.assertEqual(full[key].shape, (2, SEQ_LEN))
            self.assertEqual(short[key].shape, (2, 9))
            self.assertTrue(torch.equal(short[key], full[key][:, :9]))
            # The dropped columns are padding only.
            self.assertTrue(torch.all(full[key][:, 9:] == -1))
        self.assertTrue(torch.equal(short['images'], full['images']))

    def test_to_longest_truncates_like_to_max(self):
        batch = [example(SEQ_LEN + 4), example(3)]
        full = self.collate('to_max', [dict(x) for x in batch])
        short = self.collate('to_longest', [dict(x) for x in batch])
        self.assertEqual(short['input_ids'].shape, (2, SEQ_LEN))
        self.assertTrue(torch.equal(short['input_ids'], full['input_ids']))


if __name__ == '__main__':
    unittest.main()
