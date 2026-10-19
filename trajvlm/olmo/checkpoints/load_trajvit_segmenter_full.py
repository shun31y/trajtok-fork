"""Load the released retrieval checkpoint's vision encoder into TrajVLM."""
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor


def segmenter_state(checkpoint, expected):
    source = checkpoint['model']
    state = {k.removeprefix('vision_encoder.'): v for k, v in source.items()
             if k.startswith('vision_encoder.')}
    missing = set(expected) - set(state)
    unexpected = {k for k in set(state) - set(expected)
                  if not k.startswith(('traj_seg_head_high_res.', 'patch_decoder_high_res.'))}
    mismatched = [k for k in set(expected) & set(state) if expected[k].shape != state[k].shape]
    if missing or unexpected or mismatched:
        raise ValueError(f'Incompatible segmenter checkpoint: missing={sorted(missing)}, '
                         f'unexpected={sorted(unexpected)}, shape_mismatch={mismatched}')
    # Text encoder and unused high-resolution heads belong to segmenter training.
    return {k: state[k] for k in expected}


def load_full_segmenter(backbone, path):
    model = backbone.segmenter
    expected = model.state_dict()
    # Keep aliases: ConvNeXt exposes its final LayerNorm both as norm and
    # norms.3. Generic state-dict distribution can drop or leave plain Tensor
    # values for the second name, causing mixed Tensor/DTensor copies.
    targets = dict(model.named_parameters(remove_duplicate=False))
    targets.update(dict(model.named_buffers(remove_duplicate=False)))
    sharded = any(isinstance(targets[k], DTensor) for k in expected)
    if sharded and not dist.is_initialized():
        raise RuntimeError('Sharded segmenter initialization requires a process group')
    state, error = {}, None
    if not sharded or dist.get_rank() == 0:
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            state = segmenter_state(checkpoint, expected)
        except Exception as exc:
            if not sharded:
                raise
            error = f'{type(exc).__name__}: {exc}'
    if sharded:
        errors = [error]
        dist.broadcast_object_list(errors, src=0)
        if errors[0] is not None:
            raise ValueError(errors[0])
        # Broadcast one full tensor at a time, then apply the destination's
        # mesh/placement. This also handles ordinary buffers alongside DTensors
        # without retaining the complete checkpoint on every GPU.
        with torch.no_grad():
            for name in sorted(expected):
                target = targets[name]
                if dist.get_rank() == 0:
                    value = state[name].to(device=target.device, dtype=target.dtype).contiguous()
                else:
                    value = torch.empty(target.shape, device=target.device, dtype=target.dtype)
                dist.broadcast(value, src=0)
                if isinstance(target, DTensor):
                    value = distribute_tensor(value, target.device_mesh, target.placements,
                                              src_data_rank=None)
                target.copy_(value)
        return
    model.load_state_dict(state, strict=True)
