"""base.py - Base trainer class."""

import torch
import numpy as np
import random
from pathlib import Path
from typing import Optional, Dict, Any

from config import ExperimentConfig
from utils import set_reproducible_seeds


class BaseTrainer:
    """Base class for model training."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        val_dataset,
        config: ExperimentConfig,
        save_path: Path,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.save_path = save_path
        self.device = config.device
        self.model.to(self.device)
        
        set_reproducible_seeds(config.random_seed, deterministic=True)

    def create_dataloaders(self, shuffle_train: bool = True):
        """Create deterministic DataLoaders."""
        batch_size = self.config.training.batch_size
        seed = self.config.random_seed
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        def worker_init_fn(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        
        if shuffle_train and hasattr(self.train_dataset, 'sample_weights') and self.train_dataset.sample_weights is not None:
            weights = torch.as_tensor(self.train_dataset.sample_weights, dtype=torch.double)
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True, generator=g)
            train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=worker_init_fn,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=worker_init_fn,
                generator=g,
            )
        
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )
        
        return train_loader, val_loader

    def save_checkpoint(self, epoch: int, metrics: Dict[str, Any]):
        """Save model checkpoint."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            **metrics,
        }
        torch.save(checkpoint, self.save_path)

    def load_checkpoint(self, checkpoint_path: Path, strict: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        return checkpoint