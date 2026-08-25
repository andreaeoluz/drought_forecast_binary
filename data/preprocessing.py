"""preprocessing.py - Data preprocessing functions."""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple


# ============================================================================
# SPATIAL PROCESSING
# ============================================================================

def downsample_block_mean(data: np.ndarray, factor_h: int, factor_w: int) -> np.ndarray:
    """
    Downsample spatial resolution via block mean.
    
    Args:
        data: Input array (H, W) or (H, W, C)
        factor_h: Height downsampling factor
        factor_w: Width downsampling factor
    
    Returns:
        Downsampled array
    """
    is_2d = data.ndim == 2
    if is_2d:
        data = data[..., np.newaxis]
    
    H, W, C = data.shape
    Hc = (H // factor_h) * factor_h
    Wc = (W // factor_w) * factor_w
    data = data[:Hc, :Wc, :]
    
    data = data.reshape(Hc // factor_h, factor_h, Wc // factor_w, factor_w, C)
    result = data.mean(axis=(1, 3))
    
    return result.squeeze(-1) if is_2d else result


def build_valid_mask(
    data_stack: np.ndarray,
    min_valid_ratio: float = 0.7,
    verbose: bool = True
) -> np.ndarray:
    """
    Build mask of valid pixels based on missing data.
    
    The mask indicates whether a pixel HAS DATA (not NaN/Inf),
    regardless of the value (0, 1, or any other).
    
    Para séries temporais climáticas com p=12, precisamos que o pixel tenha
    dados em pelo menos 85% do período para representar adequadamente
    o ciclo sazonal.

    Args:
        data_stack: (T, H, W, C)
        min_valid_ratio: Minimum fraction of timesteps with finite data
        verbose: If True, prints statistics

    Returns:
        mask: (H, W) - True for pixels with valid data
    """
    T, H, W, C = data_stack.shape
    
    valid_count = np.zeros((H, W), dtype=np.float32)
    
    for t in range(T):
        for c in range(C):
            band = data_stack[t, :, :, c]
            valid = ~(np.isnan(band) | np.isinf(band))
            valid_count += valid.astype(np.float32)
    
    total_observations = T * C
    valid_ratio = valid_count / total_observations
    mask = valid_ratio >= min_valid_ratio
    
    if verbose:
        total_pixels = mask.size
        valid_pixels = mask.sum()
        invalid_pixels = total_pixels - valid_pixels
        print(f"\n📊 Validity mask (min_valid_ratio={min_valid_ratio:.2f}):")
        print(f"   Total pixels: {total_pixels:,}")
        print(f"   Valid pixels: {valid_pixels:,} ({valid_pixels/total_pixels:.1%})")
        print(f"   Invalid pixels: {invalid_pixels:,} ({invalid_pixels/total_pixels:.1%})")
        
        # Estatísticas adicionais sobre a distribuição da valid_ratio
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"\n   Valid ratio distribution:")
        for p in percentiles:
            val = np.percentile(valid_ratio[valid_ratio > 0], p)
            print(f"      {p}th percentile: {val:.3f}")
        print(f"      Mean: {valid_ratio[valid_ratio > 0].mean():.3f}")
    
    return mask


def build_temporal_valid_mask(
    data_stack: np.ndarray,
    months: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    p: int = 12,
    q: int = 1,
    mode: str = "classification"
) -> np.ndarray:
    """
    Build temporal validity mask for sequence-based models.
    
    ✅ NOVO: Verifica se CADA sequência temporal é válida.
    
    Esta função verifica se, para cada timestep t, a sequência completa
    [t, t+1, ..., t+p-1] e o target [t+p+q-1] são válidos.
    
    Args:
        data_stack: (T, H, W, C)
        months: (T,) month indices
        valid_mask: (H, W) spatial validity mask
        p: History length
        q: Forecast horizon
        mode: 'autoencoder' or 'classification'
    
    Returns:
        temporal_mask: (T,) - True for timesteps with complete sequences
    """
    T, H, W, C = data_stack.shape
    
    # Verificar quais timesteps têm dados válidos
    timestep_valid = np.ones(T, dtype=bool)
    
    # Para cada timestep, verificar se pelo menos um pixel é válido
    for t in range(T):
        if valid_mask is not None:
            # Verificar se há pixels válidos neste timestep
            for c in range(C):
                band = data_stack[t, :, :, c]
                valid_pixels = ~(np.isnan(band) | np.isinf(band))
                if valid_pixels.any():
                    break
            else:
                timestep_valid[t] = False
        else:
            # Sem máscara, verificar se todos os dados são finitos
            if not np.all(np.isfinite(data_stack[t])):
                timestep_valid[t] = False
    
    # Construir máscara temporal
    if mode == "autoencoder":
        # Autoencoder: precisa de p meses consecutivos válidos
        temporal_mask = np.zeros(T, dtype=bool)
        for t in range(T - p + 1):
            if all(timestep_valid[t + i] for i in range(p)):
                temporal_mask[t] = True
    else:
        # Classification: precisa de p meses de entrada E q meses à frente
        temporal_mask = np.zeros(T, dtype=bool)
        for t in range(T - p - q + 1):
            # Verificar entrada
            entrada_valida = all(timestep_valid[t + i] for i in range(p))
            # Verificar target
            target_idx = t + p + q - 1
            target_valido = timestep_valid[target_idx]
            
            if entrada_valida and target_valido:
                temporal_mask[t] = True
    
    valid_timesteps = temporal_mask.sum()
    print(f"\n📊 Temporal validity mask (p={p}, q={q}, mode={mode}):")
    print(f"   Total timesteps: {T}")
    print(f"   Valid sequences: {valid_timesteps} ({valid_timesteps/T:.1%})")
    
    return temporal_mask


def apply_domain_mask(
    data: np.ndarray,
    mask: np.ndarray,
    fill_value: float = np.nan
) -> np.ndarray:
    """
    Apply domain mask to data.
    
    Note: This function should NOT be used on input data for training!
    The mask should only be applied in the loss calculation.
    
    Useful for:
    - Visualization (show only valid pixels)
    - Post-processing
    - GeoTIFF generation (use appropriate nodata)

    Args:
        data: (T, H, W, C) or (H, W, C)
        mask: (H, W) - True for valid pixels
        fill_value: Value for invalid pixels

    Returns:
        Data with invalid pixels filled with fill_value
    """
    if mask is None:
        return data
    
    masked = data.copy()
    
    if data.ndim == 4:
        T, H, W, C = data.shape
        mask_expanded = mask[None, :, :, None]
        mask_expanded = np.broadcast_to(mask_expanded, (T, H, W, C))
        masked[~mask_expanded] = fill_value
    elif data.ndim == 3:
        H, W, C = data.shape
        mask_expanded = mask[:, :, None]
        mask_expanded = np.broadcast_to(mask_expanded, (H, W, C))
        masked[~mask_expanded] = fill_value
    elif data.ndim == 2:
        masked[~mask] = fill_value
    else:
        raise ValueError(f"Unsupported dimensions: {data.ndim}")
    
    return masked


# ============================================================================
# NORMALIZATION
# ============================================================================

class ClimateNormalizer:
    """
    Seasonal normalizer for climate variables.
    
    For each band and month: norm = (x - mean_month) / std_month
    
    The normalization preserves all pixels (including value 0).
    The mask is used ONLY to calculate statistics, not to modify data.
    
    ✅ CORREÇÃO: Garantir que não haja NaN/Inf após normalização
    """

    def __init__(self, bands: List[str]):
        self.bands = bands
        self.num_bands = len(bands)
        self.params: Dict[str, Dict[int, Dict[str, float]]] = {}
        self.global_stats: Dict[str, Dict[str, float]] = {}

    def fit(self, data: np.ndarray, months: np.ndarray, valid_mask: Optional[np.ndarray] = None):
        """
        Calculate mean and std per band and month.
        
        Args:
            data: (T, H, W, C)
            months: (T,) month indices
            valid_mask: (H, W) - True for valid pixels
        """
        T, H, W, C = data.shape
        
        if C != self.num_bands:
            raise ValueError(f"Expected {self.num_bands} bands, got {C}")
        
        sum_by_month = {band: {m: 0.0 for m in range(1, 13)} for band in self.bands}
        sum_sq_by_month = {band: {m: 0.0 for m in range(1, 13)} for band in self.bands}
        count_by_month = {band: {m: 0 for m in range(1, 13)} for band in self.bands}
        
        global_sum = {band: 0.0 for band in self.bands}
        global_sum_sq = {band: 0.0 for band in self.bands}
        global_count = {band: 0 for band in self.bands}
        
        if valid_mask is not None:
            pixel_mask = valid_mask.astype(bool)
        else:
            pixel_mask = np.ones((H, W), dtype=bool)
        
        for t in range(T):
            month = int(months[t])
            data_t = data[t]
            
            for b_idx, band in enumerate(self.bands):
                band_data = data_t[:, :, b_idx]
                valid_values = band_data[pixel_mask]
                valid_values = valid_values[np.isfinite(valid_values)]
                
                if len(valid_values) == 0:
                    continue
                
                sum_by_month[band][month] += valid_values.sum()
                sum_sq_by_month[band][month] += (valid_values ** 2).sum()
                count_by_month[band][month] += len(valid_values)
                
                global_sum[band] += valid_values.sum()
                global_sum_sq[band] += (valid_values ** 2).sum()
                global_count[band] += len(valid_values)
        
        for band in self.bands:
            self.params[band] = {}
            
            for month in range(1, 13):
                n = count_by_month[band][month]
                
                if n == 0:
                    if global_count[band] > 0:
                        mean = global_sum[band] / global_count[band]
                        variance = (global_sum_sq[band] / global_count[band]) - (mean ** 2)
                        std = np.sqrt(max(variance, 1e-6))
                    else:
                        mean, std = 0.0, 1.0
                else:
                    mean = sum_by_month[band][month] / n
                    variance = (sum_sq_by_month[band][month] / n) - (mean ** 2)
                    std = np.sqrt(max(variance, 1e-6))
                
                self.params[band][month] = {"mean": float(mean), "std": float(std)}
        
        for band in self.bands:
            if global_count[band] > 0:
                mean = global_sum[band] / global_count[band]
                variance = (global_sum_sq[band] / global_count[band]) - (mean ** 2)
                std = np.sqrt(max(variance, 1e-6))
                self.global_stats[band] = {"mean": float(mean), "std": float(std)}
            else:
                self.global_stats[band] = {"mean": 0.0, "std": 1.0}

    def transform(self, data: np.ndarray, months: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply normalization.
        
        ✅ CORREÇÃO: Garantir que não haja NaN/Inf após normalização
        
        Args:
            data: (T, H, W, C)
            months: (T,) month indices
            valid_mask: (H, W) - True for valid pixels (used only for statistics)
        
        Returns:
            Normalized data with no NaN/Inf
        """
        if not self.params:
            raise RuntimeError("Normalizer not fitted.")
        
        T, H, W, C = data.shape
        data_norm = np.zeros_like(data, dtype=np.float32)
        
        for t in range(T):
            month = int(months[t])
            data_t = data[t]
            
            for b_idx, band in enumerate(self.bands):
                if month not in self.params[band]:
                    mean = self.global_stats[band]["mean"]
                    std = self.global_stats[band]["std"]
                else:
                    params = self.params[band][month]
                    mean, std = params["mean"], params["std"]
                
                band_data = data_t[:, :, b_idx]
                band_norm = (band_data - mean) / std
                data_norm[t, :, :, b_idx] = band_norm
        
        # ✅ CORREÇÃO: Garantir que não haja NaN/Inf
        data_norm = np.nan_to_num(data_norm, nan=0.0)
        data_norm = np.where(np.isinf(data_norm), 0.0, data_norm)
        
        return data_norm

    def fit_transform(self, data: np.ndarray, months: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and apply normalization.
        
        ✅ CORREÇÃO: Garantir que não haja NaN/Inf após normalização
        """
        self.fit(data, months, valid_mask)
        return self.transform(data, months, valid_mask)

    def save(self, path: Path):
        """Save parameters."""
        params_serializable = {}
        for band, months_dict in self.params.items():
            params_serializable[band] = {str(m): stats for m, stats in months_dict.items()}
        
        save_data = {
            "bands": self.bands,
            "params": params_serializable,
            "global_stats": self.global_stats,
        }
        
        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ClimateNormalizer":
        """Load parameters."""
        with open(path, "r") as f:
            data = json.load(f)
        
        normalizer = cls(data["bands"])
        
        normalizer.params = {}
        for band, months_dict in data["params"].items():
            normalizer.params[band] = {int(m): stats for m, stats in months_dict.items()}
        
        normalizer.global_stats = data.get("global_stats", {})
        
        if not normalizer.global_stats:
            for band in normalizer.bands:
                means = [normalizer.params[band][m]["mean"] for m in normalizer.params[band]]
                stds = [normalizer.params[band][m]["std"] for m in normalizer.params[band]]
                if means:
                    normalizer.global_stats[band] = {
                        "mean": float(np.mean(means)),
                        "std": float(np.mean(stds))
                    }
                else:
                    normalizer.global_stats[band] = {"mean": 0.0, "std": 1.0}
        
        return normalizer


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_data_integrity(
    data: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    raise_on_error: bool = True
) -> Tuple[bool, Dict]:
    """
    Validate data integrity before training.
    
    ✅ NOVO: Verifica se há NaN/Inf nos dados e máscara.
    
    Args:
        data: (T, H, W, C)
        valid_mask: (H, W) - True for valid pixels
        raise_on_error: If True, raises ValueError on error
    
    Returns:
        (is_valid, diagnostics)
    """
    T, H, W, C = data.shape
    
    diagnostics = {
        "has_nan": False,
        "has_inf": False,
        "nan_count": 0,
        "inf_count": 0,
        "valid_pixels": 0,
        "invalid_pixels": 0,
        "message": ""
    }
    
    # Verificar NaN/Inf
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    
    diagnostics["has_nan"] = nan_count > 0
    diagnostics["has_inf"] = inf_count > 0
    diagnostics["nan_count"] = int(nan_count)
    diagnostics["inf_count"] = int(inf_count)
    
    # Verificar máscara
    if valid_mask is not None:
        valid_pixels = valid_mask.sum()
        invalid_pixels = valid_mask.size - valid_pixels
        diagnostics["valid_pixels"] = int(valid_pixels)
        diagnostics["invalid_pixels"] = int(invalid_pixels)
        
        # Verificar se há pixels com dados mas marcados como inválidos
        for t in range(min(10, T)):  # Amostrar alguns timesteps
            for c in range(C):
                band = data[t, :, :, c]
                has_data = ~(np.isnan(band) | np.isinf(band))
                # Pixels com dados mas inválidos na máscara
                inconsistent = has_data & ~valid_mask
                if inconsistent.any():
                    diagnostics["message"] = f"Inconsistent pixels: data present but mask invalid"
                    break
    
    # Mensagem de erro
    if diagnostics["has_nan"] or diagnostics["has_inf"]:
        diagnostics["message"] = (
            f"Data contains {nan_count:,} NaN and {inf_count:,} Inf values. "
            "Run nan_to_num before training."
        )
        if raise_on_error:
            raise ValueError(diagnostics["message"])
    
    print(f"\n📊 Data Integrity Check:")
    print(f"   NaN values: {nan_count:,} ({'✅' if nan_count == 0 else '❌'})")
    print(f"   Inf values: {inf_count:,} ({'✅' if inf_count == 0 else '❌'})")
    if valid_mask is not None:
        print(f"   Valid pixels: {valid_pixels:,} ({valid_pixels/valid_mask.size:.1%})")
        print(f"   Invalid pixels: {invalid_pixels:,} ({invalid_pixels/valid_mask.size:.1%})")
    
    return nan_count == 0 and inf_count == 0, diagnostics