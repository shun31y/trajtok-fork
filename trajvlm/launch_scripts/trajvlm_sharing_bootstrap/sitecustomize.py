"""Set PyTorch tensor sharing strategy in spawned DataLoader workers."""

import torch.multiprocessing as multiprocessing

multiprocessing.set_sharing_strategy("file_system")
