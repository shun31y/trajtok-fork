from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo.checkpoints.load_trajvit_segmenter_full import load_full_segmenter, segmenter_state


class AliasedSegmenter(nn.Module):
    def __init__(self, mesh):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.norm.weight = nn.Parameter(distribute_tensor(torch.zeros(4), mesh, [Shard(0)]))
        self.norm.bias = nn.Parameter(distribute_tensor(torch.zeros(4), mesh, [Shard(0)]))
        self.norms = nn.ModuleList([self.norm])
        self.register_buffer('scale', torch.zeros(1))


def distributed_worker(rank, rendezvous, checkpoint, world_size):
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh('cpu', (world_size,))
        model = nn.Linear(2, 4)
        for name, parameter in list(model.named_parameters()):
            setattr(model, name, nn.Parameter(distribute_tensor(torch.zeros_like(parameter), mesh, [Shard(0)])))
        load_full_segmenter(SimpleNamespace(segmenter=model), checkpoint)
        assert torch.equal(model.weight.full_tensor(), torch.ones(4, 2))
        assert torch.equal(model.bias.full_tensor(), torch.ones(4))
        # An error on the reading rank must reach all ranks without deadlock.
        try:
            load_full_segmenter(SimpleNamespace(segmenter=model), checkpoint + '.missing')
        except ValueError as exc:
            assert 'FileNotFoundError' in str(exc)
        else:
            raise AssertionError('Missing checkpoint should fail on every rank')
        aliased = AliasedSegmenter(mesh)
        load_full_segmenter(SimpleNamespace(segmenter=aliased), checkpoint + '.aliased')
        assert aliased.norm is aliased.norms[0]
        assert torch.equal(aliased.norms[0].weight.full_tensor(), torch.ones(4))
        assert torch.equal(aliased.norms[0].bias.full_tensor(), torch.ones(4))
        assert torch.equal(aliased.scale, torch.ones(1))
    finally:
        dist.destroy_process_group()


class SegmenterCheckpointTests(unittest.TestCase):
    def test_distributed_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'model.pth')
            torch.save({'model': {'vision_encoder.weight': torch.ones(4, 2),
                                  'vision_encoder.bias': torch.ones(4)}}, path)
            torch.save({'model': {f'vision_encoder.{name}': value for name, value in {
                'norm.weight': torch.ones(4), 'norm.bias': torch.ones(4),
                'norms.0.weight': torch.ones(4), 'norms.0.bias': torch.ones(4),
                'scale': torch.ones(1)}.items()}}, path + '.aliased')
            for world_size in (1, 2):
                with self.subTest(world_size=world_size):
                    mp.spawn(distributed_worker,
                             args=(str(Path(directory) / f'rendezvous-{world_size}'), path, world_size),
                             nprocs=world_size)

    def test_released_prefix_and_ignored_training_heads(self):
        model = nn.Linear(2, 3)
        source = {'vision_encoder.weight': torch.ones(3, 2),
                  'vision_encoder.bias': torch.ones(3),
                  'vision_encoder.patch_decoder_high_res.extra': torch.ones(1),
                  'text_encoder.extra': torch.ones(1)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            torch.save({'model': source}, path)
            load_full_segmenter(SimpleNamespace(segmenter=model), path)
        self.assertTrue(torch.equal(model.weight, torch.ones(3, 2)))
        self.assertTrue(torch.equal(model.bias, torch.ones(3)))
        with self.assertRaises(ValueError):
            segmenter_state({'model': {'vision_encoder.weight': torch.ones(4, 2)}}, model.state_dict())


if __name__ == '__main__':
    unittest.main()
