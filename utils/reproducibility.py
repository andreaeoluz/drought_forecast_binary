"""reproducibility.py - Reproducibility configuration."""

import random
import numpy as np
import torch
import os


def set_reproducible_seeds(seed: int = 42, deterministic: bool = True):
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Base random seed.
        deterministic: If True, force deterministic PyTorch operations.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Some deterministic CUDA operations require this workspace config.
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    os.environ['PYTHONHASHSEED'] = str(seed)


def get_deterministic_loader(loader, seed: int = 42):
    """
    Rebuild a DataLoader with a seeded worker_init_fn for reproducibility.

    Args:
        loader: Existing DataLoader.
        seed: Base seed for worker initialization.

    Returns:
        A new DataLoader with worker_init_fn configured.
    """
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=loader.shuffle if hasattr(loader, 'shuffle') else False,
        sampler=loader.sampler if hasattr(loader, 'sampler') else None,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last if hasattr(loader, 'drop_last') else False,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )
