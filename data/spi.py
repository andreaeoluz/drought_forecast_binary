"""spi.py - Standardized Precipitation Index computation and cache."""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from scipy.stats import gamma, norm


def compute_spi(
    precipitation: np.ndarray,
    months: np.ndarray,
    scale: int = 3,
    min_samples: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate SPI using rolling window."""
    T, H, W = precipitation.shape
    
    print(f"\n  Calculating rolling sum (scale={scale})...")
    
    acc = np.full((T, H, W), np.nan, dtype=np.float32)
    
    for i in range(H):
        for j in range(W):
            series = precipitation[:, i, j]
            series_pd = pd.Series(series)
            acc_pd = series_pd.rolling(window=scale, min_periods=1).sum()
            acc[:, i, j] = acc_pd.values.astype(np.float32)
            acc[np.isnan(precipitation[:, i, j]), i, j] = np.nan
    
    print(f"  Computing SPI by month and pixel...")
    
    spi = np.full_like(acc, np.nan, dtype=np.float32)
    stats_by_month = {month: {"n_pixels": 0, "n_empirical": 0, "n_gamma": 0} 
                      for month in range(1, 13)}
    
    for month in range(1, 13):
        month_mask = (months == month)
        month_indices = np.where(month_mask)[0]
        
        if len(month_indices) < min_samples:
            continue
        
        acc_month = acc[month_indices]
        
        for i in range(H):
            for j in range(W):
                series = acc_month[:, i, j]
                series_clean = series[np.isfinite(series)]
                
                if len(series_clean) < min_samples:
                    continue
                
                stats_by_month[month]["n_pixels"] += 1
                mean_m = series_clean.mean()
                std_m = series_clean.std()
                
                use_empirical = (std_m < 1e-6)
                
                if not use_empirical:
                    p_zero = (series_clean == 0).mean()
                    series_pos = series_clean[series_clean > 0]
                    
                    if len(series_pos) >= min_samples:
                        try:
                            shape, _, scale_g = gamma.fit(series_pos, floc=0)
                            use_empirical = False
                            stats_by_month[month]["n_gamma"] += 1
                        except Exception:
                            use_empirical = True
                            stats_by_month[month]["n_empirical"] += 1
                    else:
                        use_empirical = True
                        stats_by_month[month]["n_empirical"] += 1
                else:
                    stats_by_month[month]["n_empirical"] += 1
                
                for t_idx, t in enumerate(month_indices):
                    x = acc[t, i, j]
                    
                    if not np.isfinite(x):
                        continue
                    
                    if use_empirical:
                        if std_m > 1e-6:
                            spi[t, i, j] = (x - mean_m) / std_m
                        else:
                            spi[t, i, j] = 0.0
                    else:
                        if x == 0:
                            Hx = p_zero
                        else:
                            Gx = gamma.cdf(x, shape, 0, scale_g)
                            Hx = p_zero + (1 - p_zero) * Gx
                        
                        Hx = np.clip(Hx, 1e-6, 1 - 1e-6)
                        spi[t, i, j] = norm.ppf(Hx)
    
    spi = np.clip(spi, -6, 6)
    
    delta_spi = np.full_like(spi, np.nan, dtype=np.float32)
    delta_spi[1:] = spi[1:] - spi[:-1]
    if T > 0:
        delta_spi[0] = 0.0
    
    return spi, delta_spi


def analyze_spi_statistics(
    spi: np.ndarray,
    valid_mask: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    scale: int = 3,
) -> Dict:
    """Analyze SPI statistics."""
    T, H, W = spi.shape
    
    valid_indices = np.where(valid_mask.flatten())[0]
    n_valid_pixels = len(valid_indices)
    
    spi_reshaped = spi.reshape(T, -1)
    spi_masked = spi_reshaped[:, valid_indices]
    spi_clean = spi_masked[~np.isnan(spi_masked)]
    
    stats = {
        "scale": scale,
        "total_timesteps": T,
        "total_pixels": n_valid_pixels,
        "total_values": T * n_valid_pixels,
        "valid_values": len(spi_clean),
    }
    
    if len(spi_clean) == 0:
        return stats
    
    stats.update({
        "mean": float(np.mean(spi_clean)),
        "std": float(np.std(spi_clean)),
        "min": float(np.min(spi_clean)),
        "max": float(np.max(spi_clean)),
        "median": float(np.median(spi_clean)),
    })
    
    categories = {
        "Extreme Drought (≤ -2.0)": (spi_clean <= -2.0).sum(),
        "Severe Drought (-2.0 to -1.5)": ((spi_clean > -2.0) & (spi_clean <= -1.5)).sum(),
        "Moderate Drought (-1.5 to -1.0)": ((spi_clean > -1.5) & (spi_clean <= -1.0)).sum(),
        "Near Normal (-1.0 to 1.0)": ((spi_clean > -1.0) & (spi_clean < 1.0)).sum(),
        "Moderate Wet (1.0 to 1.5)": ((spi_clean >= 1.0) & (spi_clean < 1.5)).sum(),
        "Severe Wet (1.5 to 2.0)": ((spi_clean >= 1.5) & (spi_clean < 2.0)).sum(),
        "Extreme Wet (≥ 2.0)": (spi_clean >= 2.0).sum(),
    }
    
    stats["categories"] = {k: int(v) for k, v in categories.items()}
    stats["categories_pct"] = {k: float(100 * v / len(spi_clean)) for k, v in categories.items()}
    
    return stats


def save_spi_cache(
    spi: np.ndarray,
    delta_spi: np.ndarray,
    scale: int,
    months: np.ndarray,
    cache_dir: Path,
    stats: Dict = None,
) -> Path:
    """Save SPI to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"spi_scale_{scale}.pkl"
    
    save_data = {
        "spi": spi.astype(np.float32),
        "delta_spi": delta_spi.astype(np.float32),
        "scale": scale,
        "months": months,
    }
    
    if stats:
        save_data["statistics"] = stats
    
    with open(path, "wb") as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return path


def load_spi_cache(scale: int, cache_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load SPI from cache."""
    path = cache_dir / f"spi_scale_{scale}.pkl"
    
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    return data["spi"], data["delta_spi"]