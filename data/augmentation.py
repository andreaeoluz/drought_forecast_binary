"""augmentation.py - Extreme drought augmentation for distribution shift."""

import numpy as np
from typing import Optional, Tuple
import random


class ExtremeDroughtAugmenter:
    """
    Simple augmenter that creates more extreme drought samples.

    Purpose: Help the model generalize to test periods with more severe
    drought conditions than seen during training.

    Strategy: When a sample contains drought, amplify the drought signals
    to create more extreme versions.
    """

    def __init__(
        self,
        severity_factor: float = 0.3,
        expansion_factor: float = 0.2,
        prob: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        Initialize augmenter.

        Args:
            severity_factor: How much to intensify drought (0.1-0.5)
            expansion_factor: How much to expand drought area (0-0.3)
            prob: Probability of applying augmentation (0-1)
            seed: Random seed for reproducibility
        """
        self.severity_factor = severity_factor
        self.expansion_factor = expansion_factor
        self.prob = prob
        self._rng = random.Random(seed) if seed is not None else random
        self._applied = False

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Apply augmentation to create more extreme drought.

        Args:
            x: Input data (T, C, H, W) or (T, H, W)
            y: Binary mask (H, W) - 1 for drought, 0 for non-drought
            mask: Validity mask (H, W) - optional

        Returns:
            Augmented x, y, mask
        """
        self._applied = False

        # Skip if no drought or probability not met
        if y.sum() == 0 or self._rng.random() > self.prob:
            return x, y, mask

        # Determine data shape
        if x.ndim == 4:  # (T, C, H, W)
            return self._augment_4d(x, y, mask)
        else:  # (T, H, W) or (H, W)
            return self._augment_2d(x, y, mask)

    def _augment_4d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Augment 4D data (T, C, H, W)."""
        T, C, H, W = x.shape
        aug_x = x.copy()
        aug_y = y.copy()

        # Find drought pixels
        drought_pixels = aug_y > 0.5

        if not drought_pixels.any():
            return aug_x, aug_y, mask

        # 1. INTENSIFY DROUGHT SIGNALS
        for c in range(C):
            if c in [0, 2, 6]:  # pr, soil, tavg
                if c == 0:  # Precipitation: make it lower
                    aug_x[:, c, drought_pixels] *= (1 - self.severity_factor)
                elif c == 6:  # Temperature: make it higher
                    aug_x[:, c, drought_pixels] *= (1 + self.severity_factor * 0.5)
                elif c == 2:  # Soil moisture: make it lower
                    aug_x[:, c, drought_pixels] *= (1 - self.severity_factor * 0.7)

        # 2. EXPAND DROUGHT AREA (optional)
        if self.expansion_factor > 0 and self._rng.random() < 0.3:
            try:
                from scipy.ndimage import binary_dilation
                kernel_size = max(1, int(self.expansion_factor * min(H, W) / 10))
                if kernel_size > 0:
                    kernel = np.ones((kernel_size, kernel_size))
                    expanded = binary_dilation(drought_pixels, structure=kernel, iterations=1)
                    aug_y = expanded.astype(np.float32)
            except ImportError:
                pass

        # ✅ CORREÇÃO: Verificar se mask é numpy array e se não é None
        if mask is not None:
            # Converter para numpy se for tensor
            if hasattr(mask, 'numpy'):
                mask_np = mask.numpy()
            else:
                mask_np = mask
            
            # Aplicar máscara: apenas pixels válidos
            aug_y[~mask_np.astype(bool)] = 0

        self._applied = True
        return aug_x, aug_y, mask

    def _augment_2d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Augment 2D data (T, H, W) or (H, W)."""
        aug_y = y.copy()

        drought_pixels = aug_y > 0.5

        if self.expansion_factor > 0 and drought_pixels.any() and self._rng.random() < 0.3:
            try:
                from scipy.ndimage import binary_dilation
                H, W = y.shape
                kernel_size = max(1, int(self.expansion_factor * min(H, W) / 10))
                if kernel_size > 0:
                    kernel = np.ones((kernel_size, kernel_size))
                    expanded = binary_dilation(drought_pixels, structure=kernel, iterations=1)
                    aug_y = expanded.astype(np.float32)
            except ImportError:
                pass

        # ✅ CORREÇÃO: Verificar se mask é numpy array e se não é None
        if mask is not None:
            if hasattr(mask, 'numpy'):
                mask_np = mask.numpy()
            else:
                mask_np = mask
            aug_y[~mask_np.astype(bool)] = 0

        self._applied = True
        return x, aug_y, mask

    def was_applied(self) -> bool:
        """Return whether augmentation was applied to the last sample."""
        return self._applied

    def get_info(self) -> dict:
        """Return augmentation configuration."""
        return {
            "severity_factor": self.severity_factor,
            "expansion_factor": self.expansion_factor,
            "prob": self.prob,
        }


class NoAugmenter:
    """Dummy augmenter that returns data unchanged."""

    def __call__(self, x, y, mask=None):
        return x, y, mask

    def was_applied(self) -> bool:
        return False

    def get_info(self) -> dict:
        return {"augmentation": "none"}


def get_augmenter(
    augment_type: str = "none",
    severity_factor: float = 0.3,
    expansion_factor: float = 0.2,
    prob: float = 0.5,
    seed: Optional[int] = None,
) -> object:
    """
    Factory function to get an augmenter.

    Args:
        augment_type: 'none' or 'extreme'
        severity_factor: How much to intensify drought
        expansion_factor: How much to expand drought area
        prob: Probability of applying augmentation
        seed: Random seed

    Returns:
        Augmenter instance
    """
    if augment_type == "none":
        return NoAugmenter()

    if augment_type == "extreme":
        return ExtremeDroughtAugmenter(
            severity_factor=severity_factor,
            expansion_factor=expansion_factor,
            prob=prob,
            seed=seed,
        )

    raise ValueError(f"Unknown augment_type: {augment_type}")