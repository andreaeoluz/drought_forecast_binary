"""experiment_config.py - Configurações centralizadas do projeto."""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ============================================================================
# CONFIGURAÇÕES POR MÓDULO
# ============================================================================

@dataclass
class DataConfig:
    """Configuração dos dados e downsampling."""
    
    bands: List[str] = field(default_factory=lambda: [
        "pr", "pet", "soil", "srad", "vap", "vs", "tavg"
    ])
    
    # Downsampling adaptativo por região
    # 2x2: preserva ~85% dos eventos (Sul, Sudeste)
    # 3x3: comparação mais justa (Sul, Sudeste, Nordeste, Centro-Oeste)
    # 5x5: viabilidade computacional (Norte) - valores menores erro de memória
    downsample_config: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "Sul": (3, 3),
        "Sudeste": (3, 3),
        "Nordeste": (3, 3),
        "Centro-Oeste": (3, 3),
        "Norte": (5, 5),
    })
    
    downsample_h: int = 3      # Fallback
    downsample_w: int = 3      # Fallback
    min_valid_ratio: float = 0.7


@dataclass
class SPIConfig:
    """Configuração do SPI."""
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
    """Divisão temporal dos dados."""
    
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
    """Arquitetura do modelo."""
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    kernel_size: int = 3
    attention_dropout: float = 0.3
    use_attention: bool = True


@dataclass
class TrainingConfig:
    """Hiperparâmetros de treinamento."""
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
    
    temporal_decay: bool = False
    calibrate: bool = True
    calibrator_type: str = "platt"


@dataclass
class LossConfig:
    """Função de perda."""
    name: str = "focal"
    gamma: float = 3.0
    alpha: Optional[float] = None
    use_dynamic_alpha: bool = True


@dataclass
class ImbalanceConfig:
    """Estratégias para desbalanceamento."""
    use_weighted_sampling: bool = True
    max_pos_weight: float = 5.0


@dataclass
class AutoencoderConfig:
    p: int = 12
    train: bool = True
    use_anomalies: bool = True
    reconstruct_full_sequence: bool = True
    sequence_length: int = 12
    decay_rate: float = 0.6  
    variable_weights: List[float] = field(default_factory=lambda: [1.0] * 7)


@dataclass
class OptimizationConfig:
    """Métricas e seleção de modelos."""
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
    """Aumentação de dados para secas extremas."""
    enabled: bool = True
    augment_type: str = "extreme"
    severity_factor: float = 0.5
    expansion_factor: float = 0.2
    prob: float = 0.5


# ============================================================================
# CONFIGURAÇÃO PRINCIPAL
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuração completa do experimento."""
    
    # Região
    region: str = "Centro-Oeste"
    
    # Módulos
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
    
    # Grid search
    p_values: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    q_values: List[int] = field(default_factory=lambda: [1, 3, 6, 9, 12])
    
    # Flags
    use_transfer_learning: bool = True
    
    # Thresholds
    calibration_thresholds: List[float] = field(default_factory=lambda: [
        round(x, 2) for x in np.arange(0.05, 0.70, 0.005)
    ])
    
    # Aleatoriedade
    random_seed: int = 42
    
    # ========================================================================
    # PROPRIEDADES
    # ========================================================================
    
    @property
    def num_bands(self) -> int:
        return len(self.data.bands)
    
    @property
    def device(self) -> torch.device:
        return torch.device(self.training.device)
    
    @property
    def min_area_downsampled(self) -> int:
        factor = self.data.downsample_h
        return max(1, 10 // (factor ** 2))
    
    @property
    def optimization_metric(self) -> str:
        return self.optimization.primary_metric
    
    # ========================================================================
    # MÉTODOS
    # ========================================================================
    
    def get_downsample(self, region: str) -> Tuple[int, int]:
        """Retorna fator de downsampling para uma região."""
        return self.data.downsample_config.get(
            region,
            (self.data.downsample_h, self.data.downsample_w)
        )
    
    def get_downsample_info(self, region: str) -> Dict:
        """Retorna informações detalhadas sobre o downsampling."""
        ds_h, ds_w = self.get_downsample(region)
        
        preservation = {
            (2, 2): "~85% dos eventos preservados",
            (3, 3): "~56% dos eventos preservados",
            (4, 4): "~35% dos eventos preservados",
        }.get((ds_h, ds_w), "desconhecido")
        
        return {
            "region": region,
            "downsample_h": ds_h,
            "downsample_w": ds_w,
            "preservation_estimate": preservation,
            "area_reduction": ds_h * ds_w,
        }
    
    def get_model_config(self, model_type: str, prevalence: Optional[float] = None) -> dict:
        """Retorna configuração para criação do modelo."""
        base = {
            "input_dim": self.num_bands,
            "hidden_dims": self.model.hidden_dims,
            "kernel_size": self.model.kernel_size,
            "use_attention": self.model.use_attention,
            "attention_dropout": self.model.attention_dropout,
        }
        
        if model_type == "autoencoder":
            base["output_dim"] = self.num_bands
            base["reconstruct_full_sequence"] = self.autoencoder.reconstruct_full_sequence
            base["sequence_length"] = self.autoencoder.sequence_length
            base["decay_rate"] = self.autoencoder.decay_rate
            
        elif model_type == "predictor":
            base["output_dim"] = 1
            base["prevalence"] = prevalence or self.spi.expected_prevalence
        
        return base
    
    def get_available_metrics(self) -> List[str]:
        return ["mcc", "csi", "f1", "precision", "recall", "balanced_accuracy"]
    
    def to_dict(self) -> Dict:
        """Converte para dicionário (logging)."""
        return {
            "region": self.region,
            "num_bands": self.num_bands,
            "device": str(self.device),
            "random_seed": self.random_seed,
            "downsample": self.get_downsample(self.region),
            "use_transfer_learning": self.use_transfer_learning,
        }