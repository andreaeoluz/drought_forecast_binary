"""Data package - Data loading, preprocessing and dataset creation."""

from .loader import load_region_timeseries
from .preprocessing import downsample_block_mean, build_valid_mask, apply_domain_mask
from .preprocessing import ClimateNormalizer
from .dataset import ClimateDataset
from .spi import compute_spi, load_spi_cache, save_spi_cache, analyze_spi_statistics

__all__ = [
    "load_region_timeseries",
    "downsample_block_mean",
    "build_valid_mask",
    "apply_domain_mask",
    "ClimateNormalizer",
    "ClimateDataset",
    "compute_spi",
    "load_spi_cache",
    "save_spi_cache",
    "analyze_spi_statistics",
]