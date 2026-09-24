"""spatial.py - Spatial post-processing utilities."""

from pathlib import Path
from typing import Tuple

import numpy as np

# Re-exported from data.preprocessing so there is a single source of truth
# for these functions instead of two independent implementations.
from data.preprocessing import (  # noqa: F401
    downsample_block_mean,
    build_valid_mask,
    apply_domain_mask,
)


def get_valid_pixel_indices(
    mask: np.ndarray,
    return_flat: bool = True
) -> np.ndarray:
    """
    Return the indices of valid pixels in a boolean mask.

    Args:
        mask: Boolean mask (H, W).
        return_flat: If True, return flat indices; otherwise 2D indices.

    Returns:
        Indices of valid pixels.
    """
    if return_flat:
        return np.where(mask.flatten())[0]
    else:
        return np.where(mask)


def create_train_val_mask(
    mask: np.ndarray,
    validation_split: float = 0.2,
    seed: int = 42,
    spatial_blocks: bool = False,
    block_size: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create train/validation masks from a domain mask.

    Args:
        mask: Domain mask (H, W).
        validation_split: Fraction of pixels reserved for validation.
        seed: Random seed for reproducibility.
        spatial_blocks: If True, split by spatial blocks instead of pixels.
        block_size: Block size used when spatial_blocks is True.

    Returns:
        (train_mask, val_mask)
    """
    np.random.seed(seed)

    if spatial_blocks:
        H, W = mask.shape
        train_mask = np.zeros_like(mask, dtype=bool)
        val_mask = np.zeros_like(mask, dtype=bool)

        blocks_h = H // block_size
        blocks_w = W // block_size

        block_indices = np.arange(blocks_h * blocks_w)
        np.random.shuffle(block_indices)

        n_val_blocks = int(blocks_h * blocks_w * validation_split)
        val_blocks = set(block_indices[:n_val_blocks])

        for bi in range(blocks_h):
            for bj in range(blocks_w):
                h_start = bi * block_size
                h_end = min(h_start + block_size, H)
                w_start = bj * block_size
                w_end = min(w_start + block_size, W)

                block_idx = bi * blocks_w + bj
                if block_idx in val_blocks:
                    val_mask[h_start:h_end, w_start:w_end] = True
                else:
                    train_mask[h_start:h_end, w_start:w_end] = True

        train_mask = train_mask & mask
        val_mask = val_mask & mask

    else:
        valid_indices = get_valid_pixel_indices(mask, return_flat=True)

        n_valid = len(valid_indices)
        n_val = int(n_valid * validation_split)

        shuffled = np.random.permutation(valid_indices)
        val_indices = shuffled[:n_val]
        train_indices = shuffled[n_val:]

        train_mask = np.zeros_like(mask, dtype=bool)
        val_mask = np.zeros_like(mask, dtype=bool)

        train_mask.flat[train_indices] = True
        val_mask.flat[val_indices] = True

    return train_mask, val_mask


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than min_size."""
    from scipy import ndimage

    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask

    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    result = mask.copy()

    for i, size in enumerate(sizes):
        if size < min_size:
            result[labeled == i + 1] = 0

    return result


def remove_small_holes(mask: np.ndarray, area_threshold: int) -> np.ndarray:
    """Fill small holes inside mask regions.

    Only fills background components that are fully enclosed by foreground
    (i.e. do NOT touch the array border). Background patches touching the
    border are the actual exterior of an irregular study domain (coastline,
    region boundary), not a hole — filling them would turn small
    border-touching gaps in the valid-data mask into fabricated "drought"
    pixels right at the domain edge.
    """
    from scipy import ndimage

    inv_mask = 1 - mask
    labeled, num = ndimage.label(inv_mask)

    if num == 0:
        return mask

    # Labels touching any edge of the array are exterior, not holes.
    border_labels = set(np.unique(labeled[0, :])) | set(np.unique(labeled[-1, :]))
    border_labels |= set(np.unique(labeled[:, 0])) | set(np.unique(labeled[:, -1]))
    border_labels.discard(0)

    sizes = ndimage.sum(inv_mask, labeled, range(1, num + 1))
    result = mask.copy()

    for i, size in enumerate(sizes):
        label_id = i + 1
        if label_id in border_labels:
            continue
        if size < area_threshold:
            result[labeled == label_id] = 1

    return result


def postprocess_binary_mask(
    probs: np.ndarray,
    threshold: float,
    min_area: int = 5,
    hole_area: int = 5,
) -> np.ndarray:
    """
    Post-process a probability map into a cleaned binary mask.

    Args:
        probs: Probabilities (H, W).
        threshold: Binarization threshold.
        min_area: Minimum area (in pixels) for a component to be kept.
        hole_area: Maximum hole area (in pixels) to fill.

    Returns:
        Post-processed binary mask.
    """
    # Same convention as evaluation.metrics.find_best_threshold (probs >= thr),
    # so a pixel exactly at the threshold is treated identically everywhere
    # a threshold decision is made in this codebase.
    binary = (probs >= threshold).astype(np.uint8)

    if binary.sum() > 0:
        binary = remove_small_objects(binary, min_area)
        binary = remove_small_holes(binary, hole_area)

    return binary


def create_validation_rasters(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    output_dir: Path,
    metadata: dict,
    prefix: str = "val"
) -> None:
    """
    Save prediction/truth/probability rasters for visual inspection.

    Args:
        predictions: Predictions (H, W).
        targets: Targets (H, W).
        mask: Validity mask (H, W).
        threshold: Binarization threshold.
        output_dir: Output directory.
        metadata: Raster metadata (crs, transform).
        prefix: Filename prefix.
    """
    import rasterio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_masked = predictions.copy()
    # The probability band uses a NaN nodata value, so it needs its own
    # float array filled with NaN rather than the 255 sentinel used by the
    # uint8 binary bands.
    predictions_prob_masked = predictions.astype(np.float32).copy()
    predictions_prob_masked[~mask] = np.nan
    targets_masked = targets.copy()

    predictions_masked[~mask] = 255
    targets_masked[~mask] = 255

    binary_pred = (predictions_masked > threshold).astype(np.uint8) * 1
    binary_pred[~mask] = 255

    profile = {
        'driver': 'GTiff',
        'height': mask.shape[0],
        'width': mask.shape[1],
        'count': 1,
        'dtype': 'uint8',
        'crs': metadata.get('crs'),
        'transform': metadata.get('transform'),
        'nodata': 255,
        'compress': 'lzw',
    }

    for name, data in [
        (f'{prefix}_pred', binary_pred),
        (f'{prefix}_truth', targets_masked.astype(np.uint8)),
        (f'{prefix}_prob', predictions_prob_masked),
    ]:
        band_profile = dict(profile)
        if name.endswith('_prob'):
            band_profile['dtype'] = 'float32'
            band_profile['nodata'] = np.nan

        out_path = output_dir / f"{name}.tif"
        with rasterio.open(out_path, 'w', **band_profile) as dst:
            if name.endswith('_prob'):
                dst.write(data, 1)
            else:
                dst.write(data.astype(np.uint8), 1)

        print(f"   Saved: {out_path}")