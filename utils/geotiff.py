"""geotiff.py - Unified GeoTIFF export helpers."""

import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, Optional, Union


def save_geotiff(
    data: np.ndarray,
    metadata: Dict,
    out_path: Union[str, Path],
    dtype: Optional[str] = None,
    nodata: Optional[Union[int, float]] = None,
    compress: bool = True,
) -> Path:
    """
    Save any 2D array as a GeoTIFF, auto-detecting dtype when needed.

    Args:
        data: 2D array to save.
        metadata: Dict with 'crs', 'transform', 'height', 'width'.
        out_path: Output path.
        dtype: Output dtype ('uint8', 'float32', or 'auto').
        nodata: Nodata value (None disables it).
        compress: Whether to use LZW compression.

    Returns:
        Path of the saved file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if dtype is None or dtype == 'auto':
        if data.dtype == np.uint8 or data.dtype == bool:
            dtype_out = 'uint8'
            data_out = data.astype(np.uint8)
        else:
            dtype_out = 'float32'
            data_out = data.astype(np.float32)
    else:
        dtype_out = dtype
        data_out = data.astype(dtype)

    expected_h = metadata.get('original_height', metadata.get('height', data.shape[0]))
    expected_w = metadata.get('original_width', metadata.get('width', data.shape[1]))

    if data_out.shape != (expected_h, expected_w):
        from skimage.transform import resize
        data_out = resize(data_out, (expected_h, expected_w), order=0, preserve_range=True)
        data_out = data_out.astype(dtype_out)

    meta = {
        "driver": "GTiff",
        "height": data_out.shape[0],
        "width": data_out.shape[1],
        "count": 1,
        "dtype": dtype_out,
        "crs": metadata["crs"],
        "transform": metadata["transform"],
    }

    if nodata is not None:
        meta["nodata"] = nodata

    if compress:
        meta.update({
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        })

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data_out, 1)

    return out_path


def save_probability_geotiff(
    probs: np.ndarray,
    metadata: Dict,
    out_path: Union[str, Path],
    nodata: float = np.nan,
) -> Path:
    """Save a probability map as a float32 GeoTIFF (NaN nodata by default,
    matching the convention used elsewhere for probability rasters, e.g.
    utils.spatial.create_validation_rasters)."""
    return save_geotiff(probs, metadata, out_path, dtype='float32', nodata=nodata)


def save_binary_geotiff(
    mask: np.ndarray,
    metadata: Dict,
    out_path: Union[str, Path],
    nodata: int = 255,
) -> Path:
    """Save a binary mask as a uint8 GeoTIFF.

    nodata defaults to 255, NOT 0: the mask's valid values are 0 (no
    drought) and 1 (drought), so 0 cannot double as the nodata sentinel
    without a GIS consumer misreading every "no drought" pixel as missing
    data. 255 matches the sentinel already used for binary rasters in
    utils.spatial.create_validation_rasters.
    """
    return save_geotiff(mask, metadata, out_path, dtype='uint8', nodata=nodata)