"""experiment_config.py - Centralized project configuration."""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ============================================================================
# PER-MODULE CONFIGURATION
# ============================================================================

@dataclass
class DataConfig:
    """Data and downsampling configuration."""

    bands: List[str] = field(default_factory=lambda: [
        "pr", "pet", "soil", "srad", "vap", "vs", "tavg"
    ])

    # Region-specific downsampling factor.
    # 3x3: fairer comparison across regions (South, Southeast, Northeast, Center-West).
    # 5x5: computational feasibility for the larger North region (lower memory footprint).
    # Keys match the raw data folder names on disk and must stay in Portuguese.
    downsample_config: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "Sul": (3, 3),
        "Sudeste": (3, 3),
        "Nordeste": (3, 3),
        "Centro-Oeste": (3, 3),
        "Norte": (5, 5),
    })

    downsample_h: int = 3      # Fallback for regions not listed above
    downsample_w: int = 3      # Fallback for regions not listed above
    min_valid_ratio: float = 0.7


@dataclass
class SPIConfig:
    """SPI (Standardized Precipitation Index) configuration."""
    scale: int = 3
    threshold: float = -2.0
    threshold_name: str = "extreme"
    min_samples: int = 30

    REFERENCE_PREVALENCES = {
        "extreme": 0.029,
        "severe": 0.070,
        "moderate": 0.153,
    }

    @property
    def expected_prevalence(self) -> float:
        return self.REFERENCE_PREVALENCES.get(self.threshold_name, 0.029)


@dataclass
class SplitConfig:
    """Chronological train/validation/test split."""

    train_gs: Tuple[str, str] = ("1980-01", "2019-12")
    val_gs: Tuple[str, str] = ("2020-01", "2022-12")
    train_final: Tuple[str, str] = ("1980-01", "2022-12")
    test: Tuple[str, str] = ("2023-01", "2024-12")

    def ym_to_int(self, year_month: str) -> int:
        y, m = map(int, year_month.split('-'))
        return y * 12 + (m - 1)

    def get_end_indices_gs(self) -> Dict:
        return {
            "train_end": self.ym_to_int(self.train_gs[1]),
            "val_end": self.ym_to_int(self.val_gs[1]),
        }

    def get_indices(self) -> Dict:
        return {
            "trainval_end": self.ym_to_int(self.train_final[1]),
            "test_end": self.ym_to_int(self.test[1]),
        }

    def get_test_period_length(self) -> int:
        start = self.ym_to_int(self.test[0])
        end = self.ym_to_int(self.test[1])
        return end - start + 1


@dataclass
class ModelArchConfig:
    """Model architecture."""
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    kernel_size: int = 3
    attention_dropout: float = 0.3
    use_attention: bool = True
    attention_weight: float = 0.3

    # Multiscale temporal module (improvement doc, item 4): fast + slow
    # branches run alongside the attention context, so the model can
    # represent both sudden-onset and persistent drought development
    # without needing a longer history window `p`.
    use_multiscale: bool = True
    multiscale_fast_window: int = 3
    multiscale_weight: float = 0.3

    # Residual adapter between the fused (encoder + attention + multiscale)
    # representation and the decoder (improvement doc, item 3): lets the
    # predictor learn only the adjustments it needs, instead of overwriting
    # the pretrained representation.
    use_residual_adapter: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    epochs: int = 300
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3

    freeze_encoder_epochs: int = 5
    encoder_lr_factor: float = 0.3
    unfreeze_after: int = 5

    # Partial/progressive transfer learning (improvement doc, item 1):
    # when enabled, the encoder is unfrozen one ConvLSTM layer at a time -
    # deepest (task-specific) first, shallowest (general) last - instead of
    # all at once. `progressive_unfreeze_epoch_gap` epochs pass between
    # stages; `layer_lr_decay` shrinks the LR further for each shallower
    # layer (see PredictorTrainer._create_optimizer).
    progressive_unfreeze: bool = False
    progressive_unfreeze_epoch_gap: int = 5
    layer_lr_decay: float = 0.7

    temporal_decay: bool = False
    calibrate: bool = True
    calibrator_type: str = "platt"


@dataclass
class LossConfig:
    """Loss function configuration."""
    name: str = "focal"
    gamma: float = 3.0
    alpha: Optional[float] = None
    use_dynamic_alpha: bool = True


@dataclass
class ImbalanceConfig:
    """Class imbalance handling strategies."""
    use_weighted_sampling: bool = True
    max_pos_weight: float = 2.0


@dataclass
class AutoencoderConfig:
    p: int = 12
    train: bool = True
    use_anomalies: bool = True
    reconstruct_full_sequence: bool = True
    sequence_length: int = 12
    decay_rate: float = 0.3
    skip_weight: float = 0.1
    attention_weight: float = 0.3
    diversity_weight: float = 0.01
    variable_weights: List[float] = field(default_factory=lambda: [1.0] * 7)


@dataclass
class OptimizationConfig:
    """Metrics and model selection."""
    primary_metric: str = "mcc"
    secondary_metric: str = "csi"

    characterization_metrics: List[str] = field(default_factory=lambda: [
        "csi", "mcc", "f1", "precision", "recall"
    ])

    min_csi_threshold: float = 0.0
    min_mcc_threshold: float = 0.0

    threshold_min: float = 0.05
    threshold_max: float = 0.70
    threshold_step: float = 0.01


@dataclass
class AugmentationConfig:
    """Extreme-drought data augmentation."""
    enabled: bool = True
    augment_type: str = "extreme"
    severity_factor: float = 0.5
    expansion_factor: float = 0.2
    prob: float = 0.5


@dataclass
class RollingOriginConfig:
    """
    Rolling-origin (multi-split) temporal evaluation (improvement doc,
    items 5-6): instead of relying on a single train/val/test split, whose
    test period may (as observed for North, Center-West and Southeast) be
    climatically far more severe than anything seen in training, evaluate
    the model over several chronologically-advancing splits so both model
    selection and reported metrics reflect performance across different
    climate regimes (normal, moderate drought, severe drought) rather than
    just one.
    """
    enabled: bool = False
    n_splits: int = 3
    # Months between the end of validation in one split and the next.
    step_months: int = 24
    # Length of each split's validation window, in months.
    val_window_months: int = 36
    # Minimum months of training data required before the first split.
    min_train_months: int = 120


# ============================================================================
# MAIN CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Region (matches the raw data folder name; keep in Portuguese)
    region: str = "Sul"

    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    spi: SPIConfig = field(default_factory=SPIConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelArchConfig = field(default_factory=ModelArchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    rolling_origin: RollingOriginConfig = field(default_factory=RollingOriginConfig)

    # Grid search
    p_values: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    q_values: List[int] = field(default_factory=lambda: [1, 3, 6, 9, 12])

    # Flags
    use_transfer_learning: bool = True

    # Decision-threshold search grid used during validation-based threshold
    # optimization. Rounded to 3 decimals: rounding a 0.005-step array to 2
    # decimals collapses many values into duplicates (e.g. 0.055 and 0.06
    # can round to the same value), silently halving the intended grid
    # resolution.
    calibration_thresholds: List[float] = field(default_factory=lambda: [
        round(x, 3) for x in np.arange(0.05, 0.70, 0.005)
    ])

    # Reproducibility
    random_seed: int = 42

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def num_bands(self) -> int:
        return len(self.data.bands)

    @property
    def device(self) -> torch.device:
        return torch.device(self.training.device)

    @property
    def min_area_downsampled(self) -> int:
        # Fixed at the downsampled resolution rather than derived from an
        # original-resolution pixel count: the previous "10 original pixels"
        # target, divided by factor^2 (9 for a 3x3 downsample, 25 for North's
        # 5x5), rounded down to 1 for every region - i.e. "remove components
        # smaller than 1 pixel", a no-op, since the smallest possible
        # component is already 1 pixel. 3 pixels removes genuine single-/
        # double-pixel speckle (the only scale morphological filtering can
        # actually help with - see the false-positive connected-component
        # analysis in Section 1.5 of the paper, where >89% of false-positive
        # area sits in blobs larger than 10 pixels that this cannot and
        # should not touch) without risking real, small drought patches.
        return 3

    @property
    def optimization_metric(self) -> str:
        return self.optimization.primary_metric

    # ========================================================================
    # METHODS
    # ========================================================================

    def get_downsample(self, region: str) -> Tuple[int, int]:
        """Return the downsampling factor for a region."""
        return self.data.downsample_config.get(
            region,
            (self.data.downsample_h, self.data.downsample_w)
        )

    def get_downsample_info(self, region: str) -> Dict:
        """Return detailed downsampling information for a region."""
        ds_h, ds_w = self.get_downsample(region)

        preservation = {
            (2, 2): "~85% of events preserved",
            (3, 3): "~56% of events preserved",
            (4, 4): "~35% of events preserved",
        }.get((ds_h, ds_w), "unknown")

        return {
            "region": region,
            "downsample_h": ds_h,
            "downsample_w": ds_w,
            "preservation_estimate": preservation,
            "area_reduction": ds_h * ds_w,
        }

    def get_model_config(self, model_type: str, prevalence: Optional[float] = None) -> dict:
        """Return the constructor config for a given model type."""
        base = {
            "input_dim": self.num_bands,
            "hidden_dims": self.model.hidden_dims,
            "kernel_size": self.model.kernel_size,
            "use_attention": self.model.use_attention,
            "attention_dropout": self.model.attention_dropout,
            "attention_weight": self.model.attention_weight,
        }

        if model_type == "autoencoder":
            base["output_dim"] = self.num_bands
            base["reconstruct_full_sequence"] = self.autoencoder.reconstruct_full_sequence
            base["sequence_length"] = self.autoencoder.sequence_length
            base["decay_rate"] = self.autoencoder.decay_rate

        elif model_type == "predictor":
            base["output_dim"] = 1
            base["prevalence"] = prevalence or self.spi.expected_prevalence
            base["use_multiscale"] = self.model.use_multiscale
            base["multiscale_fast_window"] = self.model.multiscale_fast_window
            base["multiscale_weight"] = self.model.multiscale_weight
            base["use_residual_adapter"] = self.model.use_residual_adapter

        return base

    def get_available_metrics(self) -> List[str]:
        return ["mcc", "csi", "f1", "precision", "recall", "balanced_accuracy"]

    def to_dict(self) -> Dict:
        """Convert to a plain dict (for logging)."""
        return {
            "region": self.region,
            "num_bands": self.num_bands,
            "device": str(self.device),
            "random_seed": self.random_seed,
            "downsample": self.get_downsample(self.region),
            "use_transfer_learning": self.use_transfer_learning,
        }
