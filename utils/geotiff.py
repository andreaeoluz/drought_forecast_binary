"""geotiff.py - Exportação unificada para GeoTIFF"""

import numpy as np
import rasterio
from rasterio.transform import Affine
from pathlib import Path
from typing import Dict, Optional, Union
from skimage.transform import resize


def save_geotiff(
    data: np.ndarray,
    metadata: Dict,
    out_path: Union[str, Path],
    dtype: Optional[str] = None,
    nodata: Optional[Union[int, float]] = None,
    compress: bool = True,
) -> Path:
    """
    Salva qualquer array como GeoTIFF com detecção automática de dtype.
    
    Args:
        data: Array 2D a ser salvo
        metadata: Dict com 'crs', 'transform', 'height', 'width'
        out_path: Caminho de saída
        dtype: Tipo de dado ('uint8', 'float32', 'auto')
        nodata: Valor para nodata (None = auto)
        compress: Usar compressão LZW
    
    Returns:
        Path do arquivo salvo
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Detectar dtype
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
    
    # Garantir tamanho correto
    expected_h = metadata.get('original_height', metadata.get('height', data.shape[0]))
    expected_w = metadata.get('original_width', metadata.get('width', data.shape[1]))
    
    if data_out.shape != (expected_h, expected_w):
        from skimage.transform import resize
        data_out = resize(data_out, (expected_h, expected_w), order=0, preserve_range=True)
        data_out = data_out.astype(dtype_out)
    
    # Preparar metadados
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
) -> Path:
    """Salva probabilidades como GeoTIFF float32."""
    return save_geotiff(probs, metadata, out_path, dtype='float32')


def save_binary_geotiff(
    mask: np.ndarray,
    metadata: Dict,
    out_path: Union[str, Path],
    nodata: int = 0,
) -> Path:
    """Salva máscara binária como GeoTIFF uint8."""
    return save_geotiff(mask, metadata, out_path, dtype='uint8', nodata=nodata)