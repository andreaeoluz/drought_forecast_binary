"""autoencoder.py - Autoencoder training with sequence reconstruction."""

import torch
import numpy as np
from pathlib import Path

from .base import BaseTrainer
from utils import set_reproducible_seeds


class AutoencoderTrainer(BaseTrainer):
    """Trainer for the ConvLSTM Autoencoder, with sequence-reconstruction support."""

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config,
        save_path: Path,
        variable_weights: torch.Tensor,
    ):
        super().__init__(model, train_dataset, val_dataset, config, save_path)

        set_reproducible_seeds(config.random_seed, deterministic=True)

        self.variable_weights = variable_weights.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        self.loss_mask = None
        if hasattr(train_dataset, 'valid_mask') and train_dataset.valid_mask is not None:
            self.loss_mask = torch.from_numpy(train_dataset.valid_mask.astype(np.float32))
            self.loss_mask = self.loss_mask.to(self.device)

            valid_pixels = train_dataset.valid_mask.sum()
            total_pixels = train_dataset.valid_mask.size
            print(f"📊 Loss mask: {valid_pixels:,} valid pixels / {total_pixels:,} total")
        else:
            print("📊 No loss mask - all pixels will be used")

        self.train_mask = train_dataset.valid_mask
        self.val_mask = val_dataset.valid_mask if hasattr(val_dataset, 'valid_mask') else None

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            x = batch["x"].to(self.device)
            x = torch.nan_to_num(x, nan=0.0)

            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(self.device)
                if mask.dim() == 2:  # (H, W)
                    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            self.optimizer.zero_grad(set_to_none=True)

            # Delegate loss computation to the model, which applies the
            # validity mask and the exponential temporal weighting.
            loss = self.model.compute_loss(
                x,
                mask=mask,
                variable_weights=self.variable_weights
            )

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item() * x.size(0)
                n_samples += x.size(0)

        return total_loss / n_samples if n_samples > 0 else float('inf')

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(self.device)
                x = torch.nan_to_num(x, nan=0.0)

                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(self.device)
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(0).unsqueeze(0)

                loss = self.model.compute_loss(
                    x,
                    mask=mask,
                    variable_weights=self.variable_weights
                )

                if torch.isfinite(loss):
                    total_loss += loss.item() * x.size(0)
                    n_samples += x.size(0)

        return total_loss / n_samples if n_samples > 0 else float('inf')

    def train(self):
        """Main training loop with guaranteed reproducibility."""
        set_reproducible_seeds(self.config.random_seed, deterministic=True)

        if hasattr(self.config.autoencoder, 'use_anomalies') and self.config.autoencoder.use_anomalies:
            print("🔹 Training on monthly anomalies (deviations from climatology)")
        else:
            print("🔹 Training on raw values")

        train_loader, val_loader = self.create_dataloaders(shuffle_train=True)

        best_val_loss = float('inf')
        patience_counter = 0
        min_epochs = 10

        if hasattr(self.model, 'reconstruct_full_sequence') and self.model.reconstruct_full_sequence:
            print("🔹 Training with full sequence reconstruction")
            print(f"   - Sequence length: {self.model.sequence_length}")
            print(f"   - Decay rate: {self.model.decay_rate}")

        for epoch in range(self.config.training.epochs):
            epoch_seed = self.config.random_seed + epoch
            set_reproducible_seeds(epoch_seed, deterministic=True)

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, {"best_val_loss": best_val_loss})
            else:
                patience_counter += 1

            print(f"Epoch {epoch:3d} | LR {current_lr:.2e} | Train {train_loss:.4f} | Val {val_loss:.4f}", end="")
            print(" ✓" if improved else f" | ES {patience_counter}/{self.config.training.patience}")

            if epoch >= min_epochs and patience_counter >= self.config.training.patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

        if self.save_path.exists():
            checkpoint = torch.load(self.save_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            final_val_loss = self.validate(val_loader)
            print(f"✅ Final Val Loss: {final_val_loss:.4f}")
        else:
            print("⚠️ No checkpoint was saved!")

        return best_val_loss
