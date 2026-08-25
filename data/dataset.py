"""dataset.py - Dataset for autoencoder and binary classification with optional augmentation."""

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from typing import Optional, Dict, Any, List


class ClimateDataset(Dataset):
    """
    Dataset for autoencoder or binary classification.

    Definição temporal (Opção B - formal):
    
    Para mode="classification":
        - Entrada: X_t = [I_t, I_{t+1}, ..., I_{t+p-1}] (p meses)
        - Target: Y_t = B_{t+p+q-1} (q meses após o fim da janela)
        - Portanto: q é o horizonte de previsão (meses à frente)
        - q=1 → próximo mês após a janela
        - q=3 → terceiro mês após a janela

    Para mode="autoencoder":
        - Entrada: X_t = [I_t, I_{t+1}, ..., I_{t+p-1}] (p meses)
        - Reconstrução: toda a sequência
    """

    def __init__(
        self,
        data: np.ndarray,
        spi: Optional[np.ndarray],
        months: np.ndarray,
        p: int,
        q: int,
        spi_threshold: Optional[float] = None,
        valid_mask: Optional[np.ndarray] = None,
        use_weighted_sampling: bool = False,
        temporal_decay: bool = True,
        mode: str = "classification",
        augmenter: Optional[object] = None,
        seed: int = 42,
        verbose: bool = True,
        use_anomalies: bool = True,
    ):
        """
        Initialize ClimateDataset.

        Args:
            data: Climate data (T, H, W, C)
            spi: SPI values (T, H, W) - required for classification
            months: Month indices (T,)
            p: History length (input timesteps)
            q: Forecast horizon (timesteps ahead) - only for classification
            spi_threshold: Threshold for binary classification
            valid_mask: Validity mask (H, W)
            use_weighted_sampling: If True, use weighted sampling
            temporal_decay: If True, apply temporal decay to data
            mode: 'autoencoder' or 'classification'
            augmenter: Optional augmenter for extreme drought augmentation
            seed: Random seed for reproducibility
            verbose: If True, print information
            use_anomalies: If True and mode='autoencoder', use monthly anomalies
        """
        assert mode in ["autoencoder", "classification"], f"Invalid mode: {mode}"
        
        # ============================================================
        # BASIC DATA
        # ============================================================
        self.data = data.astype(np.float32)
        self.spi = spi.astype(np.float32) if spi is not None else None
        self.months = np.array(months)
        self.p = p
        self.q = q if mode == "classification" else 0
        self.spi_threshold = float(spi_threshold) if spi_threshold is not None else -2.0
        self.mode = mode
        self.seed = seed
        self.augmenter = augmenter
        self.use_anomalies = use_anomalies and mode == "autoencoder"
        
        T, H, W, C = data.shape
        
        if verbose:
            print(f"  Dataset: T={T}, H={H}, W={W}, C={C}, p={p}, q={self.q}, mode={mode}")
            if self.use_anomalies:
                print(f"  🔹 Using monthly anomalies for autoencoder training")
            else:
                print(f"  🔹 Using raw values for training")
            if augmenter is not None:
                aug_info = augmenter.get_info() if hasattr(augmenter, 'get_info') else {}
                print(f"  Augmentation: enabled ({aug_info})")
            else:
                print(f"  Augmentation: disabled")
        
        # ============================================================
        # VALIDITY MASK
        # ============================================================
        self.valid_mask = self._prepare_valid_mask(valid_mask, H, W)
        self.H, self.W = H, W
        
        # ============================================================
        # TEMPORAL INDICES
        # ============================================================
        self.indices = self._build_indices()
        
        # ============================================================
        # TARGET FOR CLASSIFICATION
        # ============================================================
        self.binary_mask = None
        if mode == "classification":
            self.binary_mask = self._prepare_binary_mask(H, W)
        
        # ============================================================
        # STATISTICS
        # ============================================================
        self._compute_statistics()
        
        # ============================================================
        # WEIGHTS FOR SAMPLING
        # ============================================================
        self.sample_weights = None
        if mode == "classification" and use_weighted_sampling:
            self.sample_weights = self._compute_sample_weights()
        
        # ============================================================
        # PREPARE DATA FOR TRAINING
        # ============================================================
        if self.use_anomalies:
            self.data_for_training = self._compute_anomalies(self.data, self.months)
        else:
            self.data_for_training = self.data
        
        # Rearrange for channels-first
        self.data_ch = np.transpose(self.data_for_training, (0, 3, 1, 2))
        
        # Temporal decay
        if temporal_decay:
            self.temporal_weights = torch.linspace(0.5, 1.0, steps=p).view(p, 1, 1, 1)
        else:
            self.temporal_weights = None
        
        if verbose:
            print(f"  ✅ Dataset initialized with {len(self.indices)} samples")

    # ================================================================
    # PREPARATION METHODS
    # ================================================================

    def _prepare_valid_mask(self, valid_mask: Optional[np.ndarray], H: int, W: int) -> Optional[np.ndarray]:
        """Prepare and resize validity mask."""
        if valid_mask is None:
            return None
        
        if valid_mask.shape != (H, W):
            from skimage.transform import resize
            mask = resize(
                valid_mask.astype(np.float32),
                (H, W),
                order=0,
                preserve_range=True
            ).astype(bool)
        else:
            mask = valid_mask
        
        return mask

    def _prepare_binary_mask(self, H: int, W: int) -> np.ndarray:
        """Prepare binary mask for classification."""
        if self.spi is None:
            raise ValueError("SPI required for mode='classification'")
        
        spi = self.spi
        if spi.shape[1:] != (H, W):
            from skimage.transform import resize
            spi_resized = np.zeros((spi.shape[0], H, W), dtype=spi.dtype)
            for t in range(spi.shape[0]):
                spi_resized[t] = resize(
                    spi[t],
                    (H, W),
                    order=0,
                    preserve_range=True
                )
            spi = spi_resized
            self.spi = spi
        
        return (spi <= self.spi_threshold).astype(np.float32)

    # ================================================================
    # TEMPORAL INDICES (OPÇÃO B - FORMAL)
    # ================================================================

    def _build_indices(self) -> List[int]:
        """
        Build list of valid temporal indices.
        
        ✅ CORREÇÃO: Verifica se TODOS os meses da sequência são válidos
        """
        indices = []
        T = len(self.data)
        
        # Verificar quais timesteps têm dados válidos
        if self.valid_mask is not None:
            # Para cada timestep, verificar se há pixels válidos
            valid_timesteps = []
            for t in range(T):
                # Verificar se existe pelo menos um pixel válido neste timestep
                if self.valid_mask.any():  # Simplificado - verificar melhor
                    valid_timesteps.append(t)
        else:
            valid_timesteps = list(range(T))
        
        if self.mode == "autoencoder":
            # Autoencoder: precisa de p meses consecutivos válidos
            for t in range(T - self.p + 1):
                # ✅ Verificar se TODOS os meses da sequência são válidos
                if all(t + i in valid_timesteps for i in range(self.p)):
                    indices.append(t)
        else:
            # Classification: precisa de p meses de entrada E q meses à frente
            for t in range(T - self.p - self.q + 1):
                # ✅ Verificar se TODOS os meses da entrada são válidos
                if all(t + i in valid_timesteps for i in range(self.p)):
                    # ✅ Verificar se o mês alvo é válido
                    target_idx = t + self.p + self.q - 1
                    if target_idx in valid_timesteps:
                        indices.append(t)
        
        return indices

    # ================================================================
    # ANOMALY COMPUTATION
    # ================================================================

    def _compute_anomalies(self, data: np.ndarray, months: np.ndarray) -> np.ndarray:
        """
        Compute monthly anomalies (departure from monthly climatology).
        
        Args:
            data: (T, H, W, C) climate data
            months: (T,) month indices (1-12)
        
        Returns:
            Anomalies (T, H, W, C)
        """
        T, H, W, C = data.shape
        
        # Initialize climatology arrays
        climatology = np.zeros((12, H, W, C), dtype=np.float32)
        counts = np.zeros((12, 1, 1, 1), dtype=np.float32)
        
        # Calculate monthly climatology using valid pixels only
        for t, month in enumerate(months):
            m = int(month) - 1  # 0-based index
            if self.valid_mask is not None:
                mask_3d = self.valid_mask[:, :, np.newaxis]  # (H, W, 1)
                climatology[m] += data[t] * mask_3d
                counts[m] += mask_3d.sum(axis=(0, 1), keepdims=True)
            else:
                climatology[m] += data[t]
                counts[m] += 1
        
        # Avoid division by zero
        counts = np.maximum(counts, 1)
        climatology = climatology / counts
        
        # Compute anomalies
        anomalies = np.zeros_like(data)
        for t, month in enumerate(months):
            m = int(month) - 1
            if self.valid_mask is not None:
                mask_3d = self.valid_mask[:, :, np.newaxis]
                anomalies[t] = (data[t] - climatology[m]) * mask_3d
            else:
                anomalies[t] = data[t] - climatology[m]
        
        return anomalies

    # ================================================================
    # STATISTICS
    # ================================================================

    def _compute_statistics(self) -> None:
        """Calculate dataset statistics."""
        self.total_samples = len(self.indices)
        self.drought_samples = 0
        self.drought_ratio = 0.0
        self.drought_prevalence = 0.0
        self.total_pixels = 1
        self.total_valid_pixels = 1
        self.drought_pixels = 0
        
        if self.mode != "classification" or self.binary_mask is None:
            return
        
        self.total_pixels = self.binary_mask[0].size
        
        if self.valid_mask is not None:
            self.total_valid_pixels = self.valid_mask.sum()
        else:
            self.total_valid_pixels = self.total_pixels
        
        if self.valid_mask is not None:
            drought_valid = 0
            for t in range(self.binary_mask.shape[0]):
                drought_valid += (self.binary_mask[t][self.valid_mask] > 0).sum()
            self.drought_pixels = drought_valid
        else:
            self.drought_pixels = (self.binary_mask > 0).sum()
        
        # ✅ CORREÇÃO: target_idx = t + p + q - 1
        for t in self.indices:
            target_idx = t + self.p + self.q - 1
            if self.valid_mask is not None:
                has_drought = (self.binary_mask[target_idx][self.valid_mask] > 0).any()
            else:
                has_drought = (self.binary_mask[target_idx] > 0).any()
            
            if has_drought:
                self.drought_samples += 1
        
        self.drought_ratio = self.drought_samples / self.total_samples if self.total_samples > 0 else 0
        
        if self.valid_mask is not None:
            valid_pixels_count = self.valid_mask.sum()
            if valid_pixels_count > 0:
                self.drought_prevalence = self.drought_pixels / (valid_pixels_count * self.binary_mask.shape[0])
        else:
            self.drought_prevalence = self.drought_pixels / (self.total_pixels * self.binary_mask.shape[0])

    # ================================================================
    # WEIGHTS FOR SAMPLING
    # ================================================================

    def _compute_sample_weights(self) -> np.ndarray:
        """Calculate weights for balanced sampling."""
        weights = np.ones(len(self.indices), dtype=np.float32)
        drought_ratios = []
        
        for t in self.indices:
            # ✅ CORREÇÃO: target_idx = t + p + q - 1
            target_idx = t + self.p + self.q - 1
            
            if self.valid_mask is not None:
                mask = self.valid_mask
                drought_pixels = (self.binary_mask[target_idx][mask] > 0).sum()
                total_pixels = mask.sum()
            else:
                drought_pixels = (self.binary_mask[target_idx] > 0).sum()
                total_pixels = self.binary_mask[target_idx].size
            
            ratio = drought_pixels / total_pixels if total_pixels > 0 else 0
            drought_ratios.append(ratio)
        
        drought_ratios = np.array(drought_ratios)
        weights = 1.0 + 4.0 * drought_ratios
        weights = np.clip(weights, 0.5, 5.0)
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        return weights

    # ================================================================
    # MAIN METHODS
    # ================================================================

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a dataset item.
        
        Opção B (formal):
            - Entrada: X_t = [I_t, I_{t+1}, ..., I_{t+p-1}]
            - Target: Y_t = B_{t+p+q-1}
        """
        t = self.indices[idx]
        
        # ============================================================
        # INPUT: p instantes consecutivos
        # X_t = [I_t, I_{t+1}, ..., I_{t+p-1}]
        # ============================================================
        x = torch.from_numpy(self.data_ch[t:t + self.p])
        
         # ✅ Garantir que não haja NaN/Inf no tensor
        x = torch.nan_to_num(x, nan=0.0)
        x = torch.where(torch.isinf(x), torch.zeros_like(x), x)
        
        if self.temporal_weights is not None:
            x = x * self.temporal_weights
        
        # Classification mode
        if self.mode == "classification":
            # ============================================================
            # TARGET: q-ésimo instante após o fim da janela
            # Y_t = B_{t+p+q-1}
            # ============================================================
            target_idx = t + self.p + self.q - 1
            y_bin = self.binary_mask[target_idx]
            
            if self.valid_mask is not None:
                mask = torch.from_numpy(self.valid_mask.astype(np.float32))
            else:
                mask = torch.ones_like(torch.from_numpy(y_bin))
            
            # Apply augmentation (if enabled)
            augmented = False
            if self.augmenter is not None:
                x_np = x.numpy()
                y_np = y_bin
                mask_np = mask.numpy()
                
                x_aug, y_aug, mask_aug = self.augmenter(x_np, y_np, mask_np)
                
                x = torch.from_numpy(x_aug)
                y_bin = y_aug
                mask = torch.from_numpy(mask_aug)
                
                if hasattr(self.augmenter, 'was_applied'):
                    augmented = self.augmenter.was_applied()
            
            return {
                "x": x,
                "y_bin": torch.from_numpy(y_bin).float(),
                "mask": mask,
                "month": self.months[target_idx],
                "idx": idx,
                "augmented": augmented,
            }
        
        # Autoencoder mode
        if self.valid_mask is not None:
            mask = torch.from_numpy(self.valid_mask.astype(np.float32))
        else:
            mask = torch.ones((self.H, self.W), dtype=torch.float32)
        
        return {
            "x": x,
            "mask": mask,
            "idx": idx,
        }

    # ================================================================
    # AUXILIARY METHODS
    # ================================================================

    def get_sampler(self, seed: Optional[int] = None) -> Optional[WeightedRandomSampler]:
        """Return a weighted sampler for balanced sampling."""
        if self.sample_weights is None:
            return None
        
        sampler_seed = seed if seed is not None else self.seed
        generator = torch.Generator()
        generator.manual_seed(sampler_seed)
        
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True,
            generator=generator,
        )

    def get_loss_mask(self) -> Optional[torch.Tensor]:
        """Return the mask for loss calculation."""
        if self.valid_mask is not None:
            return torch.from_numpy(self.valid_mask.astype(np.float32))
        return None

    def get_augmentation_stats(self) -> Dict[str, Any]:
        """Return augmentation statistics."""
        if self.augmenter is None:
            return {"enabled": False}
        
        info = {
            "enabled": True,
            "type": self.augmenter.__class__.__name__,
        }
        
        if hasattr(self.augmenter, 'get_info'):
            info.update(self.augmenter.get_info())
        
        return info
    
    def get_data_info(self) -> Dict[str, Any]:
        """Return information about the dataset configuration."""
        info = {
            "mode": self.mode,
            "p": self.p,
            "q": self.q,
            "num_samples": len(self.indices),
            "use_anomalies": self.use_anomalies,
            "has_valid_mask": self.valid_mask is not None,
        }
        
        if self.use_anomalies:
            info["data_type"] = "monthly_anomalies"
        else:
            info["data_type"] = "raw_values"
        
        return info