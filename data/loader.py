"""loader.py - Climate raster loading."""

import numpy as np
import rasterio
from rasterio.transform import Affine
from pathlib import Path
from typing import Dict, Optional, Tuple

from config import ExperimentConfig
from .preprocessing import downsample_block_mean, build_valid_mask


def load_region_timeseries(
    base_path: Path,
    config: ExperimentConfig,
) -> Dict:
    """
    Load complete time series for a region.

    O downsampling é definido ADAPTATIVAMENTE por região,
    baseado na configuração em config.data.downsample_config.

    Args:
        base_path: Base directory for rasters
        config: Experiment configuration

    Returns:
        Dict with data, years, months, metadata, valid_mask
    """
    region = config.region
    num_bands = config.num_bands
    
    # ================================================================
    # DOWNSAMPLING ADAPTATIVO POR REGIÃO
    # ================================================================
    factor_h, factor_w = config.get_downsample(region)
    
    region_path = base_path / region
    files = sorted(region_path.glob(f"{region}_*.tif"))
    
    if not files:
        raise FileNotFoundError(f"No files found in {region_path}")
    
    data_list = []
    years = []
    months = []
    metadata = None
    
    print(f"\n📂 Loading time series for region: {region}")
    print(f"   Files found: {len(files)}")
    print(f"   Downsampling: {factor_h}x{factor_w}")
    
    # Exibir informação de preservação
    ds_info = config.get_downsample_info(region)
    print(f"   {ds_info['preservation_estimate']}")
    
    for f in files:
        parts = f.stem.split("_")
        year = int(parts[-2])
        month = int(parts[-1])
        
        with rasterio.open(f) as src:
            arr = src.read().astype(np.float32)
            arr = np.transpose(arr, (1, 2, 0))
            
            if arr.shape[-1] != num_bands:
                raise ValueError(
                    f"Bands in {f.name}: {arr.shape[-1]} != {num_bands}"
                )
            
            # Aplicar downsampling específico da região
            arr_ds = downsample_block_mean(arr, factor_h, factor_w)
            
            if metadata is None:
                metadata = {
                    "crs": src.crs,
                    "transform": Affine(
                        src.transform.a * factor_w,
                        src.transform.b,
                        src.transform.c,
                        src.transform.d,
                        src.transform.e * factor_h,
                        src.transform.f
                    ),
                    "height": arr_ds.shape[0],
                    "width": arr_ds.shape[1],
                }
        
        data_list.append(arr_ds)
        years.append(year)
        months.append(month)
    
    data_stack = np.stack(data_list, axis=0)
    
    print(f"\n📊 Data loaded:")
    print(f"   Shape: {data_stack.shape}")
    print(f"   Timesteps: {data_stack.shape[0]}")
    print(f"   Height: {data_stack.shape[1]}")
    print(f"   Width: {data_stack.shape[2]}")
    print(f"   Bands: {data_stack.shape[3]}")
    
    # Build valid mask based on missing data
    valid_mask = build_valid_mask(data_stack, min_valid_ratio=config.data.min_valid_ratio)
    
    total_pixels = valid_mask.size
    valid_pixels = valid_mask.sum()
    
    print(f"\n📊 Validity mask:")
    print(f"   Total pixels: {total_pixels:,}")
    print(f"   Valid pixels: {valid_pixels:,} ({valid_pixels/total_pixels:.1%})")
    print(f"   Invalid pixels: {total_pixels - valid_pixels:,} ({(total_pixels - valid_pixels)/total_pixels:.1%})")
    
    return {
        "data": data_stack.astype(np.float32),
        "years": np.array(years),
        "months": np.array(months),
        "metadata": metadata,
        "valid_mask": valid_mask,
    }