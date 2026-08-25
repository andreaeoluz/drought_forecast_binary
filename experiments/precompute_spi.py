#!/usr/bin/env python3
"""precompute_spi.py - Standardized Precipitation Index precomputation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import Dict, Any

from config import ExperimentConfig, get_paths, get_data_path
from data import load_region_timeseries, compute_spi, save_spi_cache, analyze_spi_statistics
from utils import set_reproducible_seeds
from utils.logger import Logger, Colors


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def validate_mask(valid_mask: np.ndarray, precipitation: np.ndarray, logger: Logger) -> None:
    """Validate the validity mask before SPI computation."""
    if valid_mask is None:
        logger.warning("⚠️  No validity mask provided")
        return

    total_pixels = valid_mask.size
    valid_pixels = valid_mask.sum()
    invalid_pixels = total_pixels - valid_pixels

    logger.info(f"\n📊 Validity Mask:")
    logger.info(f"   Total pixels: {total_pixels:,}")
    logger.info(f"   Valid: {valid_pixels:,} ({valid_pixels/total_pixels:.1%})")
    logger.info(f"   Invalid: {invalid_pixels:,} ({invalid_pixels/total_pixels:.1%})")

    # Check for non-drought pixels (precipitation = 0)
    zero_pixels = (precipitation[0] == 0)
    zero_valid = zero_pixels & valid_mask

    logger.info(f"\n📊 Zero precipitation (non-drought) pixels:")
    logger.info(f"   Total: {zero_pixels.sum():,}")
    logger.info(f"   Valid in mask: {zero_valid.sum():,}")

    if zero_valid.sum() == 0 and zero_pixels.sum() > 0:
        logger.warning("\n⚠️  WARNING: No non-drought pixels are considered valid!")
        logger.warning("   SPI will be calculated ONLY on drought pixels.")
    elif zero_valid.sum() > 0:
        logger.success(f"   ✅ {zero_valid.sum():,} non-drought pixels are valid")


def display_spi_statistics(stats: Dict[str, Any], logger: Logger) -> None:
    """Display SPI statistics in a formatted table."""
    if not stats:
        return

    logger.section("📊 SPI STATISTICS")

    # Basic statistics
    logger.table({
        "Scale": f"{stats.get('scale', 3)} months",
        "Timesteps": stats.get('total_timesteps', 0),
        "Valid pixels": f"{stats.get('total_pixels', 0):,}",
        "Valid values": f"{stats.get('valid_values', 0):,}",
        "Mean": f"{stats.get('mean', 0):.4f}",
        "Std": f"{stats.get('std', 0):.4f}",
        "Min": f"{stats.get('min', 0):.4f}",
        "Max": f"{stats.get('max', 0):.4f}",
        "Median": f"{stats.get('median', 0):.4f}",
    })

    # SPI Categories
    if "categories" in stats:
        logger.section("🌡️  SPI CATEGORIES")
        categories = stats["categories"]
        total = sum(categories.values())

        print(f"  {'Category':<25} {'Count':<12} {'Percentage':<10}")
        print(f"  {'─' * 50}")

        for category, count in categories.items():
            pct = 100 * count / total if total > 0 else 0
            color = Colors.RED if "Extreme Drought" in category else \
                    Colors.YELLOW if "Severe Drought" in category else \
                    Colors.CYAN if "Moderate Drought" in category else \
                    Colors.BLUE if "Extreme Wet" in category else Colors.RESET
            print(f"  {color}{category:<25}{Colors.RESET} {count:<12,} {pct:>6.2f}%")


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main() -> None:
    """Main entry point for SPI precomputation."""
    # ------------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------------
    config = ExperimentConfig()
    logger = Logger()
    set_reproducible_seeds(config.random_seed)
    paths = get_paths(config.region)

    # ------------------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------------------
    logger.header(f"📊 SPI PRECOMPUTATION - Region: {config.region}")
    logger.info(f"   Scale: {config.spi.scale} months")
    logger.info(f"   Drought threshold: SPI ≤ {config.spi.threshold} ({config.spi.threshold_name})")
    logger.info(f"   Expected prevalence: {config.spi.expected_prevalence:.2%}")

    # ------------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------------
    base_data_path = get_data_path()
    out = load_region_timeseries(base_data_path, config)

    # Extract precipitation
    pr_index = config.data.bands.index("pr")
    precipitation = out["data"][..., pr_index]

    T, H, W = precipitation.shape
    period = f"{out['years'][0]}-{out['months'][0]:02d} to {out['years'][-1]}-{out['months'][-1]:02d}"

    logger.info(f"\n📊 Precipitation data:")
    logger.info(f"   Shape: {precipitation.shape}")
    logger.info(f"   Period: {period}")
    logger.info(f"   Total pixels: {H * W:,}")
    logger.info(f"   Total values: {T * H * W:,}")

    # ------------------------------------------------------------------------
    # VALIDATE MASK
    # ------------------------------------------------------------------------
    validate_mask(out["valid_mask"], precipitation, logger)

    # ------------------------------------------------------------------------
    # COMPUTE SPI
    # ------------------------------------------------------------------------
    logger.info(f"\n🔄 Computing SPI (scale={config.spi.scale})...")

    spi, delta_spi = compute_spi(
        precipitation=precipitation,
        months=out["months"],
        scale=config.spi.scale,
        min_samples=config.spi.min_samples,
    )

    # Apply validity mask to SPI
    if out["valid_mask"] is not None:
        spi[:, ~out["valid_mask"]] = np.nan
        delta_spi[:, ~out["valid_mask"]] = np.nan

    # Check results
    total_values = spi.size
    valid_values = (~np.isnan(spi)).sum()
    nan_values = total_values - valid_values

    logger.info(f"\n📊 SPI computation complete:")
    logger.info(f"   Total values: {total_values:,}")
    logger.info(f"   Valid values: {valid_values:,} ({valid_values/total_values:.1%})")
    logger.info(f"   NaN values: {nan_values:,} ({nan_values/total_values:.1%})")

    if valid_values == 0:
        logger.error("❌ No valid SPI values computed! Check data and mask.")
        return

    # ------------------------------------------------------------------------
    # ANALYZE STATISTICS
    # ------------------------------------------------------------------------
    logger.info(f"\n📊 Analyzing SPI statistics...")

    stats = analyze_spi_statistics(
        spi=spi,
        valid_mask=out["valid_mask"],
        years=out["years"],
        months=out["months"],
        scale=config.spi.scale,
    )

    display_spi_statistics(stats, logger)

    # ------------------------------------------------------------------------
    # SAVE TO CACHE
    # ------------------------------------------------------------------------
    logger.info(f"\n💾 Saving SPI to cache...")

    save_spi_cache(
        spi=spi,
        delta_spi=delta_spi,
        scale=config.spi.scale,
        months=out["months"],
        cache_dir=paths["spi_cache_dir"],
        stats=stats,
    )

    # ------------------------------------------------------------------------
    # COMPLETION
    # ------------------------------------------------------------------------
    logger.header("✅ SPI PRECOMPUTATION COMPLETED")

    cache_path = paths["spi_cache_dir"] / f"spi_scale_{config.spi.scale}.pkl"
    stats_path = paths["spi_cache_dir"] / f"spi_scale_{config.spi.scale}_statistics.json"

    if cache_path.exists():
        logger.success(f"✅ Cache: {cache_path} ({cache_path.stat().st_size / (1024*1024):.2f} MB)")
    if stats_path.exists():
        logger.success(f"✅ Statistics: {stats_path} ({stats_path.stat().st_size / 1024:.2f} KB)")

    # ------------------------------------------------------------------------
    # NEXT STEPS
    # ------------------------------------------------------------------------
    logger.info(f"\n💡 Next steps:")
    logger.info(f"   1. Train autoencoder: python main.py train-ae")
    logger.info(f"   2. Run grid search: python main.py grid-search")


if __name__ == "__main__":
    main()