"""analyze.py - Analysis of inference results."""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


def run_analysis(
    pred_dir: Optional[Path] = None,
    valid_mask_path: Optional[Path] = None,
    threshold: Optional[float] = None,
    generate_spatial: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    """
    Run analysis on inference results.

    Args:
        pred_dir: Directory containing prediction results
        valid_mask_path: Path to validity mask (optional)
        threshold: Decision threshold (optional)
        generate_spatial: Whether to generate spatial figures

    Returns:
        Tuple of (DataFrame with metrics, output directory)
    """
    print("\n" + "=" * 70)
    print("📊 ANALYZING INFERENCE RESULTS")
    print("=" * 70)

    # Find prediction directory if not provided
    if pred_dir is None:
        pred_dir = _find_prediction_dir()
        if pred_dir is None:
            print("❌ Prediction directory not found")
            print("   Use --pred-dir to specify the directory")
            return None, None

    print(f"\n📂 Analyzing: {pred_dir}")

    # Paths
    pred_path = pred_dir / "pred"
    truth_path = pred_dir / "truth"
    prob_path = pred_dir / "prob"

    if not pred_path.exists() or not truth_path.exists():
        print(f"❌ pred/ or truth/ not found in {pred_dir}")
        return None, None

    # Load data
    pred_stack, dates = _load_raster_stack(pred_path)
    obs_stack, _ = _load_raster_stack(truth_path)
    prob_stack, _ = _load_raster_stack(prob_path) if prob_path.exists() else (None, None)

    if pred_stack is None or obs_stack is None:
        print("❌ Error loading stacks")
        return None, None

    # Align lengths
    T = min(len(pred_stack), len(obs_stack))
    pred_stack, obs_stack = pred_stack[:T], obs_stack[:T]
    dates = dates[:T]

    if prob_stack is None:
        prob_stack = pred_stack.astype(np.float32)

    print(f"  • Timesteps: {T}")
    print(f"  • Grid: {pred_stack.shape[1]} x {pred_stack.shape[2]}")

    # Get valid mask
    sample_file = sorted(truth_path.glob("*.tif"))[0]
    valid_mask = _load_valid_mask(sample_file, valid_mask_path)
    print(f"  • Valid pixels: {valid_mask.sum():,} / {valid_mask.size:,} ({100*valid_mask.sum()/valid_mask.size:.1f}%)")

    # Get threshold
    if threshold is None:
        threshold = _get_threshold_from_metrics(pred_dir)
        if threshold is not None:
            print(f"  • Threshold from metrics.json: {threshold:.3f}")
        else:
            threshold = 0.3
            print(f"  • Using default threshold: {threshold:.3f}")

    # Compute metrics per timestep
    metrics_list = []
    drought_obs, drought_pred = [], []

    for t in range(T):
        m = _compute_metrics(obs_stack[t], pred_stack[t], valid_mask)
        metrics_list.append({
            "date": dates[t],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "csi": m["csi"],
            "far": m["far"],
            "bias": m["bias"],
            "mcc": m["mcc"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tn": m["tn"]
        })
        drought_obs.append(np.sum(obs_stack[t][valid_mask]))
        drought_pred.append(np.sum(pred_stack[t][valid_mask]))

    df = pd.DataFrame(metrics_list)

    # Save metrics
    out_dir = pred_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "metrics_timeseries.csv", index=False)
    print(f"  • Metrics saved: {out_dir / 'metrics_timeseries.csv'}")

    # Summary
    print("\n" + "=" * 70)
    print("📈 SUMMARY STATISTICS")
    print("=" * 70)
    print(f"  • CSI:     {df['csi'].mean():.4f} (±{df['csi'].std():.4f})")
    print(f"  • MCC:     {df['mcc'].mean():.4f} (±{df['mcc'].std():.4f})")
    print(f"  • F1:      {df['f1'].mean():.4f} (±{df['f1'].std():.4f})")
    print(f"  • Recall:  {df['recall'].mean():.4f} (±{df['recall'].std():.4f})")
    print(f"  • Precision:{df['precision'].mean():.4f} (±{df['precision'].std():.4f})")
    print(f"  • FAR:     {df['far'].mean():.4f}")
    print(f"  • Bias:    {df['bias'].mean():.4f}")
    print(f"\n  • Best CSI: {df['csi'].max():.4f} ({df.loc[df['csi'].idxmax(), 'date']})")
    print(f"  • Worst CSI:{df['csi'].min():.4f} ({df.loc[df['csi'].idxmin(), 'date']})")

    # Save summary
    summary = {
        "directory": str(pred_dir),
        "n_timesteps": T,
        "valid_pixels": int(valid_mask.sum()),
        "threshold": float(threshold),
        "metrics": {m: {"mean": float(df[m].mean()), "std": float(df[m].std()),
                       "max": float(df[m].max())} for m in ["csi", "mcc", "f1", "recall", "precision"]},
        "best_csi": {"date": str(df.loc[df['csi'].idxmax(), 'date']), "value": float(df['csi'].max())}
    }
    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Analysis complete! Results saved to: {out_dir}")
    print("=" * 70)

    return df, out_dir


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _find_prediction_dir() -> Optional[Path]:
    """Find prediction directory automatically."""
    base_paths = [
        Path("/home/andrea/projects/outputs"),
        Path("outputs"),
    ]

    for base in base_paths:
        if not base.exists():
            continue
        for region_dir in base.iterdir():
            if not region_dir.is_dir():
                continue
            inf_dir = region_dir / "inferences"
            if not inf_dir.exists():
                continue
            for threshold_dir in inf_dir.glob("threshold_*"):
                for p_dir in threshold_dir.glob("p*_q*"):
                    for model_dir in p_dir.glob("pretrained"):
                        test_dir = model_dir / "test_original"
                        if test_dir.exists() and (test_dir / "pred").exists():
                            return test_dir
    return None


def _load_raster_stack(folder: Path) -> Tuple[Optional[np.ndarray], list]:
    """Load all rasters in folder into a stack."""
    try:
        import rasterio
    except ImportError:
        return None, []

    files = sorted(folder.glob("*.tif"))
    if not files:
        return None, []

    stack = []
    dates = []

    for f in files:
        with rasterio.open(f) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            data = np.clip(data, 0, 1)
            stack.append(data)

        # Extract date from filename
        parts = f.stem.split("_")
        try:
            if len(parts) >= 3:
                dates.append(f"{parts[1]}-{parts[2]:02d}")
            else:
                dates.append(f.stem)
        except (ValueError, IndexError):
            dates.append(f.stem)

    return np.array(stack), dates


def _load_valid_mask(truth_file: Path, valid_mask_path: Optional[Path] = None) -> np.ndarray:
    """Load valid mask from truth raster."""
    try:
        import rasterio
    except ImportError:
        return np.ones((1, 1), dtype=bool)

    if valid_mask_path and valid_mask_path.exists():
        with rasterio.open(valid_mask_path) as src:
            return src.read(1).astype(bool)

    with rasterio.open(truth_file) as src:
        data = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            return (data != nodata) & ~np.isnan(data)
        return ~np.isnan(data)


def _get_threshold_from_metrics(base_dir: Path) -> Optional[float]:
    """Try to read threshold from metrics.json."""
    candidates = [
        base_dir.parent.parent / "metrics" / "metrics.json",
        base_dir.parent / "metrics" / "metrics.json",
        base_dir / "metrics" / "metrics.json",
    ]

    for f in candidates:
        if f and f.exists():
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    return data.get("threshold", None)
            except:
                pass
    return None


def _compute_metrics(obs: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
    """Compute binary classification metrics."""
    # Apply mask
    if valid_mask.shape != obs.shape:
        from skimage.transform import resize
        mask = resize(valid_mask.astype(np.float32), obs.shape,
                     order=0, preserve_range=True).astype(bool)
        obs, pred = obs[mask], pred[mask]
    else:
        obs, pred = obs[valid_mask], pred[valid_mask]

    # Flatten and clean
    obs_f = obs.flatten()
    pred_f = pred.flatten()
    valid = ~(np.isnan(obs_f) | np.isnan(pred_f) | np.isinf(obs_f) | np.isinf(pred_f))
    obs_f, pred_f = obs_f[valid], pred_f[valid]

    if len(obs_f) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "csi": 0.0,
                "far": 0.0, "bias": 1.0, "mcc": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}

    # Binarize
    obs_f = np.round(np.clip(obs_f, 0, 1)).astype(np.int32)
    pred_f = np.round(np.clip(pred_f, 0, 1)).astype(np.int32)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(obs_f, pred_f, labels=[0, 1]).ravel()
    eps = 1e-7

    precision = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + eps) if (precision + recall) > 0 else 0.0
    csi = tp / (tp + fp + fn + eps) if (tp + fp + fn) > 0 else 0.0
    far = fp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
    bias = (tp + fp) / (tp + fn + eps) if (tp + fn) > 0 else 1.0

    # MCC
    if (tp + fp) == 0 or (tp + fn) == 0 or (tn + fp) == 0 or (tn + fn) == 0:
        mcc = 0.0
    else:
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "csi": float(csi),
        "far": float(far),
        "bias": float(bias),
        "mcc": float(mcc),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
    }