#!/usr/bin/env python3
"""run.py - Main inference script."""

import sys
import argparse
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ExperimentConfig
from config.paths import get_paths, get_data_path
from inference.predictor import InferencePredictor
from utils import set_reproducible_seeds
from utils.logger import Logger


# =============================================================================
# UTILITIES
# =============================================================================

def convert_to_serializable(obj):
    """Convert numpy objects to JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


# =============================================================================
# CONFIGURATION
# =============================================================================

def find_best_config(
    region: str,
    threshold: float,
    optimization_metric: str = "csi",
    transfer_learning: bool = None
) -> dict:
    """Find the best configuration from the grid search results."""
    paths = get_paths(region, threshold=threshold)

    base = f"thr_{abs(threshold):.1f}"

    # GridSearch's own suffix always includes "_tl_{bool}" (see
    # GridSearch.__init__), plus "_noattn" for the attention-ablation runs.
    # transfer_learning isn't stored in config (it's a GridSearch/CLI
    # argument), so when the caller doesn't specify it we try every
    # plausible suffix instead of guessing wrong and missing real results.
    tl_options = [True, False] if transfer_learning is None else [transfer_learning]

    config_files = []
    for tl in tl_options:
        suffix = f"{base}_tl_{tl}"
        suffix_variants = [suffix, f"{suffix}_noattn"] if tl else [suffix]
        for s in suffix_variants:
            config_files.append(paths["grid_search_results"] / s / f"best_configuration_{s}.json")

    # Legacy/bare fallbacks, kept in case an older run wrote here directly.
    config_files += [
        paths["grid_search_results"] / "best_configuration.json",
        paths["grid_search_results"] / "best_model_by_csi.json",
        paths["grid_search_dir"] / "best_configuration.json",
    ]

    results_path = None
    for f in config_files:
        if f.exists():
            results_path = f
            break

    if results_path is None:
        searched = "\n".join(f"  - {c}" for c in config_files)
        raise FileNotFoundError(
            f"No configuration found for region '{region}' with threshold {threshold}. Searched:\n{searched}"
        )

    with open(results_path, "r") as f:
        data = json.load(f)

    if "best_configuration" in data:
        best = data["best_configuration"]
    else:
        best = data

    if "p" not in best or "q" not in best:
        raise ValueError("Configuration does not contain p and q")

    return best


def list_available_models(config: ExperimentConfig):
    """List available trained models for inference."""
    paths = get_paths(config.region, threshold=config.spi.threshold)
    logger = Logger()

    logger.header(f"AVAILABLE MODELS - {config.region}")

    search_dirs = []

    if "grid_search_pretrained" in paths:
        search_dirs.append(("pretrained", paths["grid_search_pretrained"]))
        # Attention-ablation runs (load_attention=False) save to a sibling
        # "pretrained_noattn" directory (see GridSearch._train_and_evaluate);
        # without this, --list-models silently omits those checkpoints.
        noattn_dir = paths["grid_search_pretrained"].parent / "pretrained_noattn"
        search_dirs.append(("pretrained_noattn", noattn_dir))
    if "grid_search_scratch" in paths:
        search_dirs.append(("scratch", paths["grid_search_scratch"]))

    gs_dir = paths.get("grid_search_dir")
    if gs_dir and gs_dir.exists():
        for model_type in ["pretrained", "scratch"]:
            d = gs_dir / model_type
            if d.exists():
                search_dirs.append((model_type, d))

    models = []
    for model_type, search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for model_file in search_dir.glob("model_*.pth"):
            name = model_file.stem
            parts = name.split("_")

            p = None
            q = None

            for part in parts:
                if part.startswith("p") and part[1:].isdigit():
                    p = int(part[1:])
                elif part.startswith("q") and part[1:].isdigit():
                    q = int(part[1:])

            if p is not None and q is not None:
                models.append({
                    "p": p,
                    "q": q,
                    "type": model_type,
                    "path": model_file,
                    "dirname": search_dir.relative_to(paths["base"]) if "base" in paths else search_dir.name
                })

    seen = set()
    unique_models = []
    for m in sorted(models, key=lambda x: (x["p"], x["q"], x["type"])):
        key = (m["p"], m["q"], m["type"])
        if key not in seen:
            seen.add(key)
            unique_models.append(m)

    if not unique_models:
        logger.warning("No models found!")
        return

    print(f"\n{'p':<6} {'q':<6} {'Type':<12} {'Directory':<40}")
    print(f"{'-'*70}")
    for m in unique_models:
        print(f"{m['p']:<6} {m['q']:<6} {m['type']:<12} {str(m['dirname']):<40}")

    print(f"\n✅ Total: {len(unique_models)} models found")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point for the inference script."""
    parser = argparse.ArgumentParser(
        description="Inference on the test period",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference with automatic best-configuration selection (RECOMMENDED)
  python3 main.py inference --region Sul --threshold-spi -2.0 --use-calibration --recalibrate

  # Inference with a specific configuration
  python3 main.py inference --region Sul --p 9 --q 1 --model-type pretrained --use-calibration

  # Inference without calibration
  python3 main.py inference --region Sul --p 9 --q 1 --no-calibration --threshold 0.30

  # List available models
  python3 main.py inference --region Sul --list-models
        """
    )

    parser.add_argument("--region", type=str, default=None,
                       help="Region (default: from config)")
    parser.add_argument("--threshold-spi", type=float, default=None,
                       help="SPI threshold (default: from config)")

    parser.add_argument("--p", type=int, help="History length (context months)")
    parser.add_argument("--q", type=int, help="Forecast horizon (months ahead)")
    parser.add_argument("--model-type", choices=["pretrained", "scratch"],
                       help="Model type (pretrained or scratch)")
    parser.add_argument("--model-dir", type=str, default=None,
                       help="Custom path to the model")

    parser.add_argument("--use-calibration", action="store_true", default=True,
                       help="Enable probability calibration [default: True]")
    parser.add_argument("--no-calibration", action="store_true",
                       help="Disable probability calibration")
    parser.add_argument("--recalibrate", action="store_true",
                       help="Recompute the threshold from validation data instead of reusing "
                            "the one fixed during grid search [default: reuse checkpoint threshold]")
    parser.add_argument("--use-validation", action="store_true", default=False,
                       help="Same as --recalibrate [default: False]")
    parser.add_argument("--save-calibrator", action="store_true",
                       help="Save the calibrator for future use")

    parser.add_argument("--threshold", type=float,
                       help="Custom threshold (if not provided, it will be optimized)")
    parser.add_argument("--optimization-metric", choices=["mcc", "csi"], default="csi",
                       help="Metric for threshold optimization [default: csi]")
    parser.add_argument("--use-transfer-learning", dest="transfer_learning",
                       action="store_true", default=None,
                       help="Restrict automatic best-configuration search to transfer-learning runs")
    parser.add_argument("--no-transfer-learning", dest="transfer_learning",
                       action="store_false",
                       help="Restrict automatic best-configuration search to scratch (no transfer learning) runs")

    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    parser.add_argument("--use-original-test", action="store_true", default=True,
                       help="Use the original test period (2022-01 to 2024-11)")

    args = parser.parse_args()

    # =====================================================================
    # CONFIGURATION
    # =====================================================================

    config = ExperimentConfig()

    if args.region:
        config.region = args.region

    if args.threshold_spi is not None:
        config.spi.threshold = args.threshold_spi

    logger = Logger()

    if args.list_models:
        list_available_models(config)
        return

    set_reproducible_seeds(config.random_seed)
    base_data_path = get_data_path()

    # =====================================================================
    # DETERMINE p, q AND model_type
    # =====================================================================

    if args.model_dir:
        p = args.p if args.p else 6
        q = args.q if args.q else 1
        model_type = args.model_type if args.model_type else "pretrained"
        logger.info(f"📁 Using custom model: {args.model_dir}")

    elif args.p is None or args.q is None:
        logger.info("🔍 Searching for the best configuration...")
        try:
            best_config = find_best_config(
                config.region,
                config.spi.threshold,
                args.optimization_metric,
                transfer_learning=args.transfer_learning,
            )
            p = best_config["p"]
            q = best_config["q"]
            logger.success(f"✅ Best configuration: p={p}, q={q}")

            if args.model_type is None:
                model_type = "pretrained" if best_config.get("transfer_learning", False) else "scratch"
            else:
                model_type = args.model_type

            logger.info(f"  Model type: {model_type}")

        except Exception as e:
            logger.error(f"Error finding the best configuration: {e}")
            logger.info("Use --p and --q to specify manually")
            return
    else:
        p = args.p
        q = args.q
        model_type = args.model_type if args.model_type else "pretrained"

    # =====================================================================
    # CALIBRATION SETTINGS
    # =====================================================================

    if args.no_calibration:
        use_calibration = False
    else:
        use_calibration = args.use_calibration

    use_validation_fallback = args.recalibrate or args.use_validation

    logger.info(f"  Calibration: {'✅ ENABLED' if use_calibration else '❌ DISABLED'}")
    logger.info(f"  Validation fallback: {'✅ ENABLED' if use_validation_fallback else '❌ DISABLED'}")

    # =====================================================================
    # CREATE PREDICTOR
    # =====================================================================

    try:
        predictor = InferencePredictor(
            config=config,
            base_data_path=base_data_path,
            p=p,
            q=q,
            model_type=model_type,
            optimization_metric=args.optimization_metric,
            use_calibration=use_calibration,
            use_validation_fallback=use_validation_fallback,
            fixed_threshold=args.threshold,
            model_dir=Path(args.model_dir) if args.model_dir else None,
        )
    except Exception as e:
        logger.error(f"❌ Error creating predictor: {e}")
        return

    # =====================================================================
    # RUN INFERENCE
    # =====================================================================

    result = predictor.run_inference()

    if not result or not result.get("success", True):
        logger.error("❌ Inference failed!")
        return

    # =====================================================================
    # SAVE RASTERS
    # =====================================================================

    predictor.save_rasters(result)

    # =====================================================================
    # SAVE METRICS
    # =====================================================================

    import pandas as pd

    metrics_path = predictor.paths.get("analysis_metrics", predictor.paths["inference_dir"] / "metrics")
    metrics_path.mkdir(parents=True, exist_ok=True)

    metrics_data = {
        "region": config.region,
        "spi_threshold": config.spi.threshold,
        "p": p,
        "q": q,
        "model_type": model_type,
        "optimization_metric": args.optimization_metric,
        "threshold": result["threshold"],
        "n_samples": result["n_samples"],
        "used_fallback": result.get("used_fallback", False),
        "calibration_used": use_calibration and predictor.calibrator is not None,
    }

    if result.get("metrics") is not None:
        for key, value in result["metrics"].items():
            metrics_data[f"metrics_{key}"] = value

    if result.get("calibration_metrics") is not None:
        for key, value in result["calibration_metrics"].items():
            metrics_data[f"calibration_{key}"] = value

    df = pd.DataFrame([metrics_data])
    excel_file = metrics_path / "metrics.xlsx"
    df.to_excel(excel_file, index=False)
    logger.success(f"✅ Metrics saved to: {excel_file}")

    json_file = metrics_path / "metrics.json"
    metrics_data_serializable = convert_to_serializable(metrics_data)
    with open(json_file, "w") as f:
        json.dump(metrics_data_serializable, f, indent=2)
    logger.success(f"✅ Metrics JSON saved to: {json_file}")

    if args.save_calibrator and predictor.calibrator is not None:
        import joblib
        calib_path = metrics_path / "calibrator.pkl"
        joblib.dump(predictor.calibrator, calib_path)
        logger.success(f"✅ Calibrator saved to: {calib_path}")

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================

    logger.header("📊 INFERENCE SUMMARY")
    logger.info(f"  Region: {config.region}")
    logger.info(f"  SPI Threshold: {config.spi.threshold}")
    logger.info(f"  p: {p}, q: {q}")
    logger.info(f"  Model: {model_type}")
    logger.info(f"  Period: {config.split.test[0]} to {config.split.test[1]}")
    logger.info(f"  Samples: {result['n_samples']}")
    logger.info(f"  Threshold: {result['threshold']:.3f}")
    logger.info(f"  Calibration: {'✅ Enabled' if (use_calibration and predictor.calibrator is not None) else '❌ Disabled'}")
    logger.info(f"  CSI: {result['metrics']['csi']:.4f}")
    logger.info(f"  MCC: {result['metrics']['mcc']:.4f}")
    logger.info(f"  Precision: {result['metrics']['precision']:.4f}")
    logger.info(f"  Recall: {result['metrics']['recall']:.4f}")
    logger.info(f"  F1: {result['metrics']['f1']:.4f}")

    if result.get("used_fallback", False):
        logger.info("  Threshold source: recalibrated on validation data (--recalibrate)")
    else:
        logger.info("  Threshold source: checkpoint (fixed on validation during grid search)")

    logger.info(f"\n📁 Rasters saved to: {predictor.paths['inference_dir']}")


if __name__ == "__main__":
    main()