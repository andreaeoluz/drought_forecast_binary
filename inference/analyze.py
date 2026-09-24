"""analyze.py - Analysis of inference results.

Computes per-timestep classification metrics (precision, recall, F1, CSI,
FAR, bias, MCC), spatial diagnostics (frequency, bias, confusion, CSI maps),
a calibration curve (if probability rasters are available), and timeseries
plots -- all written under a run's "analysis/" directory.

Usage (build the path from region/p/q/model-type, matching config.paths):
    python -m inference.analyze --region Sul --p 3 --q 1 --model-type pretrained

Usage (point directly at a "test_original"-style directory containing
pred/, truth/, and optionally prob/):
    python -m inference.analyze --pred-dir outputs/Sul/inferences/threshold_1.5/p3_q1/pretrained/test_original
"""

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
        pred_dir: Directory containing prediction results.
        valid_mask_path: Path to a validity mask (optional).
        threshold: Decision threshold (optional).
        generate_spatial: Whether to generate spatial figures.

    Returns:
        Tuple of (metrics DataFrame, output directory).
    """
    print("\n" + "=" * 70)
    print("ANALYZING INFERENCE RESULTS")
    print("=" * 70)

    if pred_dir is None:
        pred_dir = _find_prediction_dir()
        if pred_dir is None:
            print("Prediction directory not found")
            print("   Use --pred-dir, or --region/--p/--q/--model-type, to specify it")
            return None, None

    print(f"\nAnalyzing: {pred_dir}")

    pred_path = pred_dir / "pred"
    truth_path = pred_dir / "truth"
    prob_path = pred_dir / "prob"

    if not pred_path.exists() or not truth_path.exists():
        print(f"pred/ or truth/ not found in {pred_dir}")
        return None, None

    pred_stack, dates = _load_raster_stack(pred_path)
    obs_stack, _ = _load_raster_stack(truth_path)
    prob_stack, _ = _load_raster_stack(prob_path) if prob_path.exists() else (None, None)

    if pred_stack is None or obs_stack is None:
        print("Error loading stacks")
        return None, None

    T = min(len(pred_stack), len(obs_stack))
    pred_stack, obs_stack = pred_stack[:T], obs_stack[:T]
    dates = dates[:T]

    has_prob = prob_stack is not None and len(prob_stack) >= T
    if has_prob:
        prob_stack = prob_stack[:T]
    else:
        prob_stack = pred_stack.astype(np.float32)

    print(f"  Timesteps: {T}")
    print(f"  Grid: {pred_stack.shape[1]} x {pred_stack.shape[2]}")

    sample_file = sorted(truth_path.glob("*.tif"))[0]
    valid_mask = _load_valid_mask(sample_file, valid_mask_path)
    print(f"  Valid pixels: {valid_mask.sum():,} / {valid_mask.size:,} "
          f"({100 * valid_mask.sum() / valid_mask.size:.1f}%)")

    if threshold is None:
        threshold = _get_threshold_from_metrics(pred_dir)
        if threshold is not None:
            print(f"  Threshold from metrics.json: {threshold:.3f}")
        else:
            threshold = 0.3
            print(f"  Using default threshold: {threshold:.3f}")

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
        drought_obs.append(float(np.sum(obs_stack[t][valid_mask])))
        drought_pred.append(float(np.sum(pred_stack[t][valid_mask])))

    df = pd.DataFrame(metrics_list)

    out_dir = pred_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "metrics_timeseries.csv", index=False)
    print(f"  Metrics saved: {out_dir / 'metrics_timeseries.csv'}")

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"  CSI:       {df['csi'].mean():.4f} (+/-{df['csi'].std():.4f})")
    print(f"  MCC:       {df['mcc'].mean():.4f} (+/-{df['mcc'].std():.4f})")
    print(f"  F1:        {df['f1'].mean():.4f} (+/-{df['f1'].std():.4f})")
    print(f"  Recall:    {df['recall'].mean():.4f} (+/-{df['recall'].std():.4f})")
    print(f"  Precision: {df['precision'].mean():.4f} (+/-{df['precision'].std():.4f})")
    print(f"  FAR:       {df['far'].mean():.4f}")
    print(f"  Bias:      {df['bias'].mean():.4f}")
    print(f"\n  Best CSI:  {df['csi'].max():.4f} ({df.loc[df['csi'].idxmax(), 'date']})")
    print(f"  Worst CSI: {df['csi'].min():.4f} ({df.loc[df['csi'].idxmin(), 'date']})")

    summary: Dict[str, Any] = {
        "directory": str(pred_dir),
        "n_timesteps": T,
        "valid_pixels": int(valid_mask.sum()),
        "threshold": float(threshold),
        "metrics": {m: {"mean": float(df[m].mean()), "std": float(df[m].std()),
                       "max": float(df[m].max())} for m in ["csi", "mcc", "f1", "recall", "precision"]},
        "best_csi": {"date": str(df.loc[df['csi'].idxmax(), 'date']), "value": float(df['csi'].max())},
        "worst_csi": {"date": str(df.loc[df['csi'].idxmin(), 'date']), "value": float(df['csi'].min())},
    }

    # ------------------------------------------------------------------
    # Spatial diagnostics, GeoTIFF exports, and figures
    # ------------------------------------------------------------------
    if generate_spatial:
        try:
            lon, lat = _get_geo_coords(sample_file)
        except Exception as exc:
            print(f"  (skipping spatial outputs: could not read geo-coordinates: {exc})")
            lon = lat = None

        if lon is not None:
            freq_pred = np.zeros_like(pred_stack[0], dtype=np.float64)
            freq_obs = np.zeros_like(obs_stack[0], dtype=np.float64)
            tp_map = np.zeros_like(pred_stack[0], dtype=np.float32)
            fp_map = np.zeros_like(pred_stack[0], dtype=np.float32)
            fn_map = np.zeros_like(pred_stack[0], dtype=np.float32)

            for t in range(T):
                p_t, o_t = pred_stack[t], obs_stack[t]
                freq_pred[valid_mask] += p_t[valid_mask]
                freq_obs[valid_mask] += o_t[valid_mask]
                tp_map[valid_mask] += (p_t[valid_mask] == 1) & (o_t[valid_mask] == 1)
                fp_map[valid_mask] += (p_t[valid_mask] == 1) & (o_t[valid_mask] == 0)
                fn_map[valid_mask] += (p_t[valid_mask] == 0) & (o_t[valid_mask] == 1)

            freq_pred /= T
            freq_obs /= T
            bias_map = freq_pred - freq_obs
            csi_map = tp_map / (tp_map + fp_map + fn_map + 1e-6)

            print("  Saving spatial GeoTIFFs and figures...")
            _save_map_as_geotiff(freq_obs, sample_file, out_dir / "freq_observed.tif")
            _save_map_as_geotiff(freq_pred, sample_file, out_dir / "freq_predicted.tif")
            _save_map_as_geotiff(bias_map, sample_file, out_dir / "bias_map.tif")
            _save_map_as_geotiff(csi_map, sample_file, out_dir / "csi_map.tif")

            _plot_drought_area_timeseries(dates, drought_obs, drought_pred,
                                          out_dir / "drought_area_timeseries.png")
            _plot_spatial_frequency_maps(freq_obs, freq_pred, lon, lat,
                                        out_dir / "spatial_frequency_maps.png")
            _plot_bias_map(bias_map, lon, lat, out_dir / "bias_map.png")
            _plot_confusion_maps(tp_map, fp_map, fn_map, lon, lat, out_dir / "confusion_maps.png")
            _plot_spatial_csi(csi_map, lon, lat, out_dir / "spatial_csi.png")
            _plot_skill_timeseries(df, out_dir / "skill_timeseries.png")
            _plot_top_predictions(obs_stack, pred_stack, dates, df,
                                 out_dir / "top3_best_predictions.png", n_top=3)
            _plot_worst_predictions(obs_stack, pred_stack, dates, df,
                                   out_dir / "top3_worst_predictions.png", n_worst=3)

        if has_prob:
            print("  Computing calibration metrics...")
            probs_valid = prob_stack[:, valid_mask].reshape(-1)
            targets_valid = obs_stack[:, valid_mask].reshape(-1)

            ece, bin_centers, bin_accuracies, bin_confidences, bin_counts = _compute_calibration_metrics(
                probs_valid, targets_valid, n_bins=10
            )

            if ece is not None:
                print(f"  ECE (Expected Calibration Error): {ece:.4f}")
                summary["calibration_ece"] = float(ece)

                cal_df = pd.DataFrame({
                    "bin_center": bin_centers,
                    "accuracy": bin_accuracies,
                    "confidence": bin_confidences,
                    "count": bin_counts
                })
                cal_df.to_csv(out_dir / "calibration_curve.csv", index=False)
                _plot_calibration_curve(
                    ece, bin_centers, bin_accuracies, bin_confidences, bin_counts,
                    out_dir / "calibration_curve.png"
                )
        else:
            print("  (no prob/ rasters found -- skipping calibration analysis)")

    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAnalysis complete! Results saved to: {out_dir}")
    print("=" * 70)

    return df, out_dir


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _find_prediction_dir() -> Optional[Path]:
    """Auto-discover a prediction directory under the project's outputs/ root.

    Assumes the script is run from the project root (as documented in the
    README: `python -m inference.analyze ...`), matching every other
    relative-path assumption already made elsewhere in this codebase.
    """
    base_paths = [
        Path("outputs"),
    ]

    for base in base_paths:
        if not base.exists():
            continue
        for region_dir in sorted(base.iterdir()):
            if not region_dir.is_dir():
                continue
            inf_dir = region_dir / "inferences"
            if not inf_dir.exists():
                continue
            for threshold_dir in sorted(inf_dir.glob("threshold_*")):
                for p_dir in sorted(threshold_dir.glob("p*_q*")):
                    for model_type in ("pretrained", "scratch"):
                        model_dir = p_dir / model_type
                        test_dir = model_dir / "test_original"
                        if test_dir.exists() and (test_dir / "pred").exists():
                            return test_dir
    return None


def _resolve_pred_dir(region: str, p: int, q: int, model_type: str,
                      threshold: Optional[float] = None, split: str = "test_original") -> Path:
    """Build the prediction directory for a given (region, p, q, model_type),
    using the same layout as config.paths.get_paths."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.paths import get_paths

    paths = get_paths(region, threshold=threshold, p=p, q=q, model_type=model_type, split=split)
    return paths["inference_dir"]


def _load_raster_stack(folder: Path) -> Tuple[Optional[np.ndarray], list]:
    """Load all rasters in a folder into a single stack, sorted by filename."""
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

        # Filenames follow "<prefix>_<year>_<month>.tif" (month zero-padded).
        parts = f.stem.split("_")
        if len(parts) >= 3:
            dates.append(f"{parts[1]}-{parts[2]}")
        else:
            dates.append(f.stem)

    return np.array(stack), dates


def _load_valid_mask(truth_file: Path, valid_mask_path: Optional[Path] = None) -> np.ndarray:
    """Load the validity mask, either from a dedicated file or from a truth raster's nodata."""
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


def _get_geo_coords(sample_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Extract longitude/latitude grids (height x width) from a GeoTIFF's transform."""
    import rasterio

    with rasterio.open(sample_file) as src:
        transform = src.transform
        width = src.width
        height = src.height

        left = transform[2]
        top = transform[5]
        right = left + transform[0] * width
        bottom = top + transform[4] * height

        lon_1d = np.linspace(left, right, width)
        lat_1d = np.linspace(top, bottom, height)
        lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

    return lon_grid, lat_grid


def _save_map_as_geotiff(data: np.ndarray, reference_file: Path, output_path: Path) -> None:
    """Save a 2D array as a GeoTIFF, using a reference file for metadata."""
    import rasterio

    with rasterio.open(reference_file) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def _get_threshold_from_metrics(base_dir: Path) -> Optional[float]:
    """Try to read the decision threshold from a saved metrics.json."""
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
            except Exception:
                pass
    return None


def _compute_metrics(obs: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
    """Compute binary classification metrics for one timestep.

    Delegates the actual confusion-matrix/metric math to
    evaluation.metrics.compute_metrics_from_arrays, the same
    carefully-guarded implementation used everywhere else in this codebase
    (grid search, threshold optimization, inference), instead of an
    independent reimplementation that could silently drift out of sync
    with it (e.g. a different MCC degenerate-case guard, no final [-1, 1]
    clip).
    """
    if valid_mask.shape != obs.shape:
        from skimage.transform import resize
        mask = resize(valid_mask.astype(np.float32), obs.shape,
                     order=0, preserve_range=True).astype(bool)
        obs, pred = obs[mask], pred[mask]
    else:
        obs, pred = obs[valid_mask], pred[valid_mask]

    obs_f = obs.flatten()
    pred_f = pred.flatten()
    valid = ~(np.isnan(obs_f) | np.isnan(pred_f) | np.isinf(obs_f) | np.isinf(pred_f))
    obs_f, pred_f = obs_f[valid], pred_f[valid]

    if len(obs_f) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "csi": 0.0,
                "far": 0.0, "bias": 1.0, "mcc": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}

    obs_f = np.round(np.clip(obs_f, 0, 1)).astype(np.int32)
    pred_f = np.round(np.clip(pred_f, 0, 1)).astype(np.int32)

    from evaluation.metrics import compute_metrics_from_arrays
    m = compute_metrics_from_arrays(pred_f, obs_f)

    return {
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "csi": m["csi"],
        "far": m["far"],
        "bias": m["bias"],
        "mcc": m["mcc"],
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
    }


def _compute_calibration_metrics(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10):
    """Compute calibration metrics: ECE (Expected Calibration Error) and reliability curve."""
    valid = ~(np.isnan(probs) | np.isnan(targets))
    probs, targets = probs[valid], targets[valid]

    if len(probs) == 0:
        return None, None, None, None, None

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_accuracies, bin_confidences, bin_counts = [], [], [], []
    ece = 0.0

    for i in range(n_bins):
        in_bin = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_confidence = float(np.mean(probs[in_bin]))
            bin_accuracy = float(np.mean(targets[in_bin]))
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)
            ece += abs(bin_accuracy - bin_confidence) * (bin_count / len(probs))

    return ece, bin_centers, bin_accuracies, bin_confidences, bin_counts


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def _plot_drought_area_timeseries(dates, obs_area, pred_area, out_path: Path) -> None:
    """Plot observed vs. predicted drought area over time."""
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error

    plt.figure(figsize=(14, 5))
    plt.plot(dates, obs_area, label="Observed", marker='o', markersize=3, linewidth=1)
    plt.plot(dates, pred_area, label="Predicted", marker='s', markersize=3, linewidth=1)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.xlabel("Date")
    plt.ylabel("Drought area (pixels)")
    plt.title("Drought Area Time Series")
    plt.legend()
    plt.grid(alpha=0.3)

    if len(obs_area) > 1 and len(pred_area) > 1:
        corr, _ = pearsonr(obs_area, pred_area)
        rmse = np.sqrt(mean_squared_error(obs_area, pred_area))
        plt.text(0.02, 0.98, f"Corr: {corr:.3f}\nRMSE: {rmse:.0f}",
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_spatial_frequency_maps(obs_freq, pred_freq, lon, lat, out_path: Path) -> None:
    """Plot observed and predicted drought frequency maps."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmin = min(obs_freq.min(), pred_freq.min())
    vmax = max(obs_freq.max(), pred_freq.max())

    for ax, data, title in zip(axes, [obs_freq, pred_freq],
                               ["Observed drought frequency", "Predicted drought frequency"]):
        im = ax.imshow(data, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                       origin="upper", cmap="Reds", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.3, linestyle="--")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Frequency")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_bias_map(bias_map, lon, lat, out_path: Path) -> None:
    """Plot the spatial bias map."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    vmax = max(abs(bias_map.min()), abs(bias_map.max()))
    im = plt.imshow(bias_map, cmap="RdBu", vmin=-vmax, vmax=vmax,
                    extent=[lon.min(), lon.max(), lat.min(), lat.max()], origin="upper")
    plt.title("Spatial prediction bias (Predicted - Observed)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(alpha=0.3, linestyle="--")
    plt.colorbar(im, fraction=0.046, label="Bias")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_confusion_maps(tp, fp, fn, lon, lat, out_path: Path) -> None:
    """Plot confusion maps (TP, FP, FN)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["True Positives", "False Positives", "False Negatives"]
    maps = [tp, fp, fn]
    cmaps = ["Greens", "Oranges", "Reds"]

    for ax, data, title, cmap in zip(axes, maps, titles, cmaps):
        im = ax.imshow(data, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                       origin="upper", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.3, linestyle="--")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_spatial_csi(csi_map, lon, lat, out_path: Path) -> None:
    """Plot the spatial CSI map."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    im = plt.imshow(csi_map, vmin=0, vmax=1,
                    extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                    origin="upper", cmap="viridis")
    plt.title("Spatial Critical Success Index (CSI)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(alpha=0.3, linestyle="--")
    plt.colorbar(im, fraction=0.046, label="CSI")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_skill_timeseries(df_metrics: pd.DataFrame, out_path: Path) -> None:
    """Plot the temporal evolution of forecast skill metrics."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(df_metrics.index, df_metrics.csi, label="CSI", marker='o', markersize=3)
    axes[0].plot(df_metrics.index, df_metrics.f1, label="F1", marker='s', markersize=3)
    axes[0].plot(df_metrics.index, df_metrics.recall, label="Recall", marker='^', markersize=3)
    axes[0].plot(df_metrics.index, df_metrics.precision, label="Precision", marker='d', markersize=3)
    axes[0].set_xticks(range(len(df_metrics.date)))
    axes[0].set_xticklabels(df_metrics.date, rotation=45, ha='right')
    axes[0].set_ylabel("Score")
    axes[0].set_title("Forecast Skill Metrics")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1)

    axes[1].plot(df_metrics.index, df_metrics.mcc, label="MCC", marker='x', markersize=3, color='purple')
    axes[1].plot(df_metrics.index, df_metrics.bias, label="Bias", marker='*', markersize=3, color='orange')
    axes[1].plot(df_metrics.index, df_metrics.far, label="FAR", marker='+', markersize=3, color='red')
    axes[1].set_xticks(range(len(df_metrics.date)))
    axes[1].set_xticklabels(df_metrics.date, rotation=45, ha='right')
    axes[1].set_ylabel("Score")
    axes[1].set_title("Additional Metrics (MCC, Bias, FAR)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_calibration_curve(ece, bin_centers, bin_accuracies, bin_confidences, bin_counts,
                            out_path: Path) -> None:
    """Plot the reliability diagram (calibration curve)."""
    import matplotlib.pyplot as plt

    if not bin_centers:
        return

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect calibration", alpha=0.7)
    plt.plot(bin_confidences, bin_accuracies, 'bo-', label="Model", markersize=8)

    total_samples = sum(bin_counts)
    for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts):
        size = count / total_samples
        plt.plot([conf, conf], [acc, 0], 'gray', alpha=0.3, linewidth=size * 20)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Reliability Diagram (ECE = {ece:.4f})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_top_predictions(obs_stack, pred_stack, dates, df_metrics: pd.DataFrame,
                          out_path: Path, n_top: int = 3) -> None:
    """Plot the top N best predictions."""
    import matplotlib.pyplot as plt

    top_n = df_metrics.nlargest(n_top, "csi")
    n_plots = len(top_n)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(n_plots, 2, figsize=(10, 4 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)

    for i, row in enumerate(top_n.itertuples()):
        idx = dates.index(row.date) if row.date in dates else None
        if idx is None:
            continue

        axes[i, 0].imshow(obs_stack[idx], cmap="Reds", interpolation="nearest", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Observed drought\n{row.date}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_stack[idx], cmap="Reds", interpolation="nearest", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Predicted drought\n{row.date} (CSI={row.csi:.3f}, MCC={row.mcc:.3f})")
        axes[i, 1].axis("off")

    plt.suptitle("Top Best Predictions", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_worst_predictions(obs_stack, pred_stack, dates, df_metrics: pd.DataFrame,
                            out_path: Path, n_worst: int = 3) -> None:
    """Plot the N worst predictions (lowest CSI where drought occurred)."""
    import matplotlib.pyplot as plt

    df_with_drought = df_metrics[df_metrics.recall > 0].copy()
    if len(df_with_drought) == 0:
        return

    worst_n = df_with_drought.nsmallest(n_worst, "csi")
    n_plots = len(worst_n)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(n_plots, 2, figsize=(10, 4 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)

    for i, row in enumerate(worst_n.itertuples()):
        idx = dates.index(row.date) if row.date in dates else None
        if idx is None:
            continue

        axes[i, 0].imshow(obs_stack[idx], cmap="Reds", interpolation="nearest", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Observed drought\n{row.date}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_stack[idx], cmap="Reds", interpolation="nearest", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Predicted drought\n{row.date} (CSI={row.csi:.3f})")
        axes[i, 1].axis("off")

    plt.suptitle("Worst Missed Predictions (Drought occurred but low CSI)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze inference results (metrics, spatial maps, calibration)."
    )
    parser.add_argument("--pred-dir", type=str, default=None,
                       help="Directory containing pred/, truth/ (and optionally prob/). "
                            "If omitted, use --region/--p/--q/--model-type instead.")
    parser.add_argument("--region", type=str, default=None, help="Region (e.g. Sul, Norte)")
    parser.add_argument("--p", type=int, default=None, help="Historical window length")
    parser.add_argument("--q", type=int, default=None, help="Forecast horizon")
    parser.add_argument("--model-type", type=str, default=None, choices=["pretrained", "scratch"],
                       help="Model type")
    parser.add_argument("--threshold-spi", type=float, default=-1.5,
                       help="SPI severity threshold used for this run [default: -1.5, 'severe', "
                            "matching SPIConfig.threshold]; only used to locate the output folder "
                            "(as 'threshold_<abs(value):.1f>'), not the decision threshold. Note "
                            "config.paths.get_paths() itself falls back to 'threshold_2.0' when no "
                            "threshold is given, which does NOT match this project's actual default "
                            "-- pass this flag explicitly if a run used a non-default SPI threshold.")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Decision threshold override (default: read from metrics.json, else 0.3)")
    parser.add_argument("--valid-mask", type=str, default=None,
                       help="Path to a validity mask GeoTIFF (optional)")
    parser.add_argument("--no-spatial", action="store_true",
                       help="Skip spatial maps/GeoTIFFs and calibration analysis (metrics CSV only)")

    args = parser.parse_args()

    if args.pred_dir is not None:
        resolved_dir = Path(args.pred_dir)
    elif args.region and args.p is not None and args.q is not None and args.model_type:
        resolved_dir = _resolve_pred_dir(
            args.region, args.p, args.q, args.model_type, threshold=args.threshold_spi
        )
    else:
        resolved_dir = None  # falls back to auto-discovery inside run_analysis

    run_analysis(
        pred_dir=resolved_dir,
        valid_mask_path=Path(args.valid_mask) if args.valid_mask else None,
        threshold=args.threshold,
        generate_spatial=not args.no_spatial,
    )