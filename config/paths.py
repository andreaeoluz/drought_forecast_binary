"""paths.py - Project path management."""

import sys
from pathlib import Path
from typing import Dict, Optional, List, Union

IS_WINDOWS = sys.platform == "win32"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if IS_WINDOWS:
    DATA_BASE_PATH = Path("C:/Users/unesp/Desktop/andrea/projects/rasters")
else:
    DATA_BASE_PATH = Path("/home/andrea/projects/rasters")


def get_data_path() -> Path:
    """Return the data directory path."""
    return DATA_BASE_PATH


def get_output_path() -> Path:
    """Return the output directory path."""
    return PROJECT_ROOT / "outputs"


def get_paths(
    region: str,
    threshold: Optional[float] = None,
    p: Optional[int] = None,
    q: Optional[int] = None,
    model_type: Optional[str] = None,
    split: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Return organized paths for the region and experiment.

    Args:
        region: Region name (Norte, Nordeste, etc.)
        threshold: SPI threshold
        p: History length
        q: Forecast horizon
        model_type: 'pretrained' or 'scratch'
        split: 'test_original' or 'validation_gs'

    Returns:
        Dictionary with configured paths
    """
    output_dir = get_output_path() / region
    threshold_str = f"threshold_{abs(threshold):.1f}" if threshold is not None else "threshold_2.0"

    paths: Dict[str, Optional[Path]] = {
        "base": output_dir,
        "output_dir": output_dir,
        "autoencoder_dir": output_dir / "autoencoder",
        "spi_cache_dir": output_dir / "spi_cache",
        "analysis_dir": output_dir / "analysis",
        "grid_search_dir": output_dir / "grid_search",
        "grid_search_results": output_dir / "grid_search" / "results",
    }

    # Grid search model directories
    grid_base = output_dir / "grid_search" / threshold_str / "models"
    paths["grid_search_pretrained"] = grid_base / "pretrained"
    paths["grid_search_scratch"] = grid_base / "scratch"

    # Inference directories (only if p and q are provided)
    if p is not None and q is not None and model_type is not None:
        inf_dir = output_dir / "inferences" / threshold_str / f"p{p}_q{q}" / model_type
        split_name = split if split else "test_original"
        paths["inference_dir"] = inf_dir / split_name
        paths["inference_pred"] = paths["inference_dir"] / "pred"
        paths["inference_truth"] = paths["inference_dir"] / "truth"
        paths["inference_prob"] = paths["inference_dir"] / "prob"
        
        analysis_dir = output_dir / "analysis" / threshold_str / f"p{p}_q{q}" / model_type
        paths["analysis_metrics"] = analysis_dir / "metrics"
        paths["analysis_spatial"] = analysis_dir / "spatial"
        paths["analysis_examples"] = analysis_dir / "examples"

    # Create essential directories
    essential_keys = [
        "base", "output_dir", "autoencoder_dir", "spi_cache_dir",
        "analysis_dir", "grid_search_dir", "grid_search_results",
        "grid_search_pretrained", "grid_search_scratch"
    ]
    
    for key in essential_keys:
        if key in paths and paths[key] is not None:
            paths[key].mkdir(parents=True, exist_ok=True)

    # Create analysis subdirectories if they exist
    for key in ["analysis_metrics", "analysis_spatial", "analysis_examples"]:
        if key in paths and paths[key] is not None:
            paths[key].mkdir(parents=True, exist_ok=True)

    paths["models_dir"] = paths["autoencoder_dir"]
    paths["results_dir"] = paths["grid_search_results"]
    paths["analysis"] = paths["analysis_dir"]

    return paths  # type: ignore


def find_model_path(
    region: str,
    p: int,
    q: int,
    model_type: str,
    threshold: Optional[float] = None,
) -> Optional[Path]:
    """
    Find the path of a specific model.

    Args:
        region: Region name
        p: History length
        q: Forecast horizon
        model_type: 'pretrained' or 'scratch'
        threshold: SPI threshold

    Returns:
        Path to the model or None if not found
    """
    paths = get_paths(region, threshold=threshold)
    model_filename = f"model_p{p}_q{q}.pth"

    candidates = []

    # Grid search directories
    if model_type == "pretrained":
        key = "grid_search_pretrained"
    else:
        key = "grid_search_scratch"
    
    if key in paths and paths[key] is not None:
        candidates.append(paths[key] / model_filename)

    # Direct grid search
    gs_dir = paths.get("grid_search_dir")
    if gs_dir is not None:
        candidates.append(gs_dir / model_type / model_filename)

    # Autoencoder directory
    ae_dir = paths.get("autoencoder_dir")
    if ae_dir is not None:
        candidates.append(ae_dir / model_filename)

    for path in candidates:
        if path is not None and path.exists():
            return path

    return None