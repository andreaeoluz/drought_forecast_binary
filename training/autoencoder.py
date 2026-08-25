"""train_autoencoder.py - Treinamento do autoencoder com reconstrução de sequência"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import random
import sys

# Adicionar o diretório pai ao path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.trainer import BaseTrainer
from utils import set_reproducible_seeds


class WeightedSmoothL1Loss(nn.Module):
    """Smooth L1 Loss com pesos por variável."""
    
    def __init__(self, variable_weights: torch.Tensor, beta: float = 0.1):
        super().__init__()
        self.beta = beta
        self.register_buffer("weights", variable_weights)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        weighted_loss = loss * self.weights.view(1, -1, 1, 1)
        return weighted_loss.mean()


class AutoencoderTrainer(BaseTrainer):
    """Trainer para ConvLSTM Autoencoder com suporte a reconstrução de sequência."""
    
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
        
        variable_weights = variable_weights.to(self.device)
        self.variable_weights = variable_weights
        self.criterion = WeightedSmoothL1Loss(variable_weights, beta=0.1)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # ============================================================
        # ✅ CORREÇÃO: Usar valid_mask em vez de original_valid_mask
        # ============================================================
        self.loss_mask = None
        if hasattr(train_dataset, 'valid_mask') and train_dataset.valid_mask is not None:
            self.loss_mask = torch.from_numpy(train_dataset.valid_mask.astype(np.float32))
            self.loss_mask = self.loss_mask.to(self.device)
            
            valid_pixels = train_dataset.valid_mask.sum()
            total_pixels = train_dataset.valid_mask.size
            print(f"📊 Máscara de perda: {valid_pixels:,} pixels válidos / {total_pixels:,} total")
        else:
            print("📊 Sem máscara de perda - todos os pixels serão usados")
        
        # ✅ Também armazenar para referência
        self.train_mask = train_dataset.valid_mask
        self.val_mask = val_dataset.valid_mask if hasattr(val_dataset, 'valid_mask') else None
    
    def train_epoch(self, train_loader):
        """Treina uma época."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        
        for batch in train_loader:
            x = batch["x"].to(self.device)
            x = torch.nan_to_num(x, nan=0.0)
            
            # ✅ Obter máscara do batch (se disponível)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(self.device)
                # Expandir para (B, T, H, W) ou (B, H, W) conforme necessário
                if mask.dim() == 2:  # (H, W)
                    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Usar o método compute_loss do modelo
            if hasattr(self.model, 'compute_loss'):
                loss = self.model.compute_loss(
                    x, 
                    mask=mask,  # ✅ Usar máscara do batch
                    variable_weights=self.variable_weights
                )
            else:
                # Comportamento original (fallback)
                target = x[:, -1].clone()
                target = torch.nan_to_num(target, nan=0.0)
                recon, _ = self.model(x)
                recon = torch.nan_to_num(recon, nan=0.0)
                loss = self.criterion(recon, target)
                
                # ✅ Aplicar máscara na perda
                if mask is not None:
                    # Máscara: True para pixels válidos
                    loss = (loss * mask).sum() / (mask.sum() + 1e-8)
                elif self.loss_mask is not None:
                    loss = (loss * self.loss_mask.unsqueeze(0).unsqueeze(0)).sum() / (self.loss_mask.sum() + 1e-8)
                else:
                    loss = loss.mean()
            
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item() * x.size(0)
                n_samples += x.size(0)
        
        return total_loss / n_samples if n_samples > 0 else float('inf')
    
    def validate(self, val_loader):
        """Valida o modelo."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(self.device)
                x = torch.nan_to_num(x, nan=0.0)
                
                # ✅ Obter máscara do batch (se disponível)
                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(self.device)
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(0).unsqueeze(0)
                
                if hasattr(self.model, 'compute_loss'):
                    loss = self.model.compute_loss(
                        x,
                        mask=mask,
                        variable_weights=self.variable_weights
                    )
                else:
                    target = x[:, -1].clone()
                    target = torch.nan_to_num(target, nan=0.0)
                    recon, _ = self.model(x)
                    recon = torch.nan_to_num(recon, nan=0.0)
                    loss = self.criterion(recon, target)
                    
                    if mask is not None:
                        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
                    elif self.loss_mask is not None:
                        loss = (loss * self.loss_mask.unsqueeze(0).unsqueeze(0)).sum() / (self.loss_mask.sum() + 1e-8)
                    else:
                        loss = loss.mean()
                
                if torch.isfinite(loss):
                    total_loss += loss.item() * x.size(0)
                    n_samples += x.size(0)
        
        return total_loss / n_samples if n_samples > 0 else float('inf')
    
    def train(self):
        """
        Loop principal de treinamento com reprodutibilidade garantida.
        """
        set_reproducible_seeds(self.config.random_seed, deterministic=True)
        
        # Verificar se está usando anomalias
        if hasattr(self.config.autoencoder, 'use_anomalies') and self.config.autoencoder.use_anomalies:
            print(f"🔹 Treinando com anomalias mensais (desvios da climatologia)")
        else:
            print(f"🔹 Treinando com valores absolutos")
        
        train_loader, val_loader = self.create_dataloaders(shuffle_train=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        min_epochs = 10
        
        # Verificar se está usando reconstrução de sequência
        if hasattr(self.model, 'reconstruct_full_sequence') and self.model.reconstruct_full_sequence:
            print(f"🔹 Treinando com reconstrução de sequência completa")
            print(f"   - Tamanho da sequência: {self.model.sequence_length}")
            print(f"   - Taxa de decaimento: {self.model.decay_rate}")
        
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
            print("⚠️ Nenhum checkpoint salvo!")
        
        return best_val_loss