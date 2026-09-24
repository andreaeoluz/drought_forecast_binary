"""threshold.py - Test-time threshold recalibration."""

import torch
import numpy as np
import json
from pathlib import Path

from config import ExperimentConfig, get_paths
from data import load_region_timeseries, ClimateNormalizer, load_spi_cache
from models import ConvLSTMPredictor
from evaluation.metrics import find_best_threshold


class ThresholdOptimizer:
    """Decision threshold optimizer for the test period."""

    def __init__(
        self,
        config: ExperimentConfig,
        base_data_path: Path,
        optimization_metric: str = None,
    ):
        self.config = config
        self.base_data_path = base_data_path
        self.threshold = config.spi.threshold
        self.paths = get_paths(config.region, threshold=self.threshold)
        self.use_transfer_learning = config.use_transfer_learning

        # Mirror GridSearch/PredictorTrainer: respect the experiment's
        # configured primary metric instead of silently hardcoding MCC.
        self.optimization_metric = (
            optimization_metric or config.optimization.primary_metric
        )

    def _candidate_suffixes(self) -> list:
        """
        Reconstruct GridSearch's own `self.suffix` values.

        `load_attention` (the attention-transfer ablation) is a constructor
        argument to GridSearch, not part of `config`, so it can't be known
        ahead of time here. When transfer learning is enabled we therefore
        try both the normal and the "_noattn" ablation suffix, in the order
        a fresh run would most likely have used.
        """
        base = f"thr_{abs(self.threshold):.1f}_tl_{self.use_transfer_learning}"
        if self.use_transfer_learning:
            return [base, base + "_noattn"]
        return [base]

    def load_best_config(self) -> dict:
        """Load the best configuration found by the grid search."""
        candidates = []
        for suffix in self._candidate_suffixes():
            results_dir = self.paths["grid_search_results"] / suffix
            candidates.append(results_dir / f"best_configuration_{suffix}.json")

        # Legacy/bare fallbacks, kept in case an older run wrote here.
        candidates += [
            self.paths["grid_search_results"] / "best_configuration.json",
            self.paths["grid_search_results"] / "best_model_by_csi.json",
            self.paths["grid_search_dir"] / "best_configuration.json",
        ]

        best_path = None
        for p in candidates:
            if p.exists():
                best_path = p
                break

        if best_path is None:
            searched = "\n".join(f"  - {c}" for c in candidates)
            raise FileNotFoundError(
                f"Configuration not found. Searched:\n{searched}"
            )

        with open(best_path, "r") as f:
            data = json.load(f)

        if "best_configuration" in data:
            return data["best_configuration"]
        return data

    def load_model(self, best_config: dict) -> ConvLSTMPredictor:
        """Load the best model checkpoint."""
        p = best_config["p"]
        q = best_config["q"]
        use_tl = best_config.get("transfer_learning", True)
        # GridSearch records this per-result (see _train_and_evaluate), so
        # we can reproduce its exact save path instead of guessing.
        load_attention = best_config.get("load_attention", True)

        model_type = "pretrained" if use_tl else "scratch"
        if use_tl:
            attn_suffix = "" if load_attention else "_noattn"
            model_dir = self.paths["grid_search_pretrained"].parent / f"pretrained{attn_suffix}"
        else:
            model_dir = self.paths["grid_search_scratch"]
        checkpoint_path = model_dir / f"model_p{p}_q{q}.pth"

        if not checkpoint_path.exists():
            checkpoint_path = self.paths["grid_search_dir"] / model_type / f"model_p{p}_q{q}.pth"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"📦 Loading model: {checkpoint_path}")

        model = ConvLSTMPredictor(self.config.get_model_config("predictor")).to(self.config.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()

        return model

    def get_test_indices(self, data_len: int, p: int, q: int) -> list:
        """Return valid sample indices for the test data."""
        # target_idx = t + q - 1 (see optimize()), matching the p/q
        # convention used by ClimateDataset: the input window ends at
        # t-1, and the target is the qth month after that window. The
        # last usable t is data_len - q (giving target_idx = data_len - 1).
        min_required = p + q
        if data_len <= min_required:
            return []
        return list(range(p, data_len - q + 1))

    def optimize(self, use_validation_fallback: bool = True):
        """Run threshold optimization."""
        print("\n" + "=" * 70)
        print("🔧 THRESHOLD OPTIMIZER")
        print("=" * 70)

        best_config = self.load_best_config()
        model = self.load_model(best_config)

        p = best_config["p"]
        q = best_config["q"]

        print(f"  Region: {self.config.region}")
        print(f"  SPI Threshold: {self.config.spi.threshold}")
        print(f"  p={p}, q={q}")

        out = load_region_timeseries(self.base_data_path, self.config)
        spi, _ = load_spi_cache(self.config.spi.scale, self.paths["spi_cache_dir"])

        years = out["years"]
        months = out["months"]
        time_idx = np.array([y * 12 + (m - 1) for y, m in zip(years, months)])

        split = self.config.split
        test_start = split.ym_to_int(split.test[0])
        test_end = split.ym_to_int(split.test[1])
        val_start = split.ym_to_int(split.train_gs[1]) + 1
        val_end = split.ym_to_int(split.val_gs[1])

        test_mask = (time_idx >= test_start) & (time_idx <= test_end)
        val_mask = (time_idx >= val_start) & (time_idx <= val_end)

        test_len = test_mask.sum()
        min_required = p + q + 1

        if test_len >= min_required:
            print(f"\n✅ Test OK: {test_len} months")
            eval_mask = test_mask
            split_name = "test"
        elif use_validation_fallback:
            val_len = val_mask.sum()
            print(f"\n⚠️ Test too short ({test_len} < {min_required})")
            print(f"   Using validation ({val_len} months) for calibration...")
            eval_mask = val_mask
            split_name = "validation"
        else:
            raise RuntimeError(f"Insufficient test data ({test_len} < {min_required})")

        data_eval = out["data"][eval_mask]
        months_eval = months[eval_mask]
        spi_eval = spi[eval_mask] if spi is not None else None
        valid_mask = out["valid_mask"]

        # Must match the normalizer this specific model was trained with -
        # pretrained/TL models were fine-tuned on the autoencoder's own
        # normalizer, scratch models on one fit fresh for the classification
        # task (see GridSearch.load_data). The two legitimately differ.
        use_tl = best_config.get("transfer_learning", True)
        if use_tl:
            candidates = [
                self.paths["autoencoder_dir"] / "normalizer.json",
                self.paths["grid_search_dir"] / "normalizer_pretrained.json",
            ]
        else:
            candidates = [
                self.paths["grid_search_dir"] / "normalizer_scratch.json",
                self.paths["autoencoder_dir"] / "normalizer.json",  # legacy fallback
            ]

        normalizer_path = next((p for p in candidates if p.exists()), None)
        if normalizer_path is None:
            raise FileNotFoundError(f"Normalizer not found among: {candidates}")

        normalizer = ClimateNormalizer.load(normalizer_path)
        print(f"✅ Normalizer loaded from: {normalizer_path}")

        data_norm = normalizer.transform(data_eval, months=months_eval, valid_mask=valid_mask)
        data_norm = np.nan_to_num(data_norm, nan=0.0)

        data_ch = np.transpose(data_norm, (0, 3, 1, 2))

        indices = self.get_test_indices(len(data_ch), p, q)

        if not indices:
            print("⚠️ No samples available!")
            return None, None

        binary_mask = (spi_eval <= self.config.spi.threshold).astype(np.float32)

        print(f"\n🔮 Collecting predictions over {len(indices)} samples ({split_name})...")

        all_probs = []
        all_targets = []

        mask_tensor = None
        if valid_mask is not None:
            mask_tensor = torch.from_numpy(valid_mask.astype(np.float32)).to(self.config.device)

        with torch.no_grad():
            for t in indices:
                target_idx = t + q - 1
                x_seq = data_ch[t - p:t]
                x_tensor = torch.from_numpy(x_seq).float().unsqueeze(0).to(self.config.device)

                logits, _ = model(x_tensor, mask=mask_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

                target = binary_mask[target_idx]

                all_probs.append(probs.flatten())
                all_targets.append(target.flatten())

        probs = np.concatenate(all_probs)
        targets = np.concatenate(all_targets)

        # The validity mask is used only to FILTER pixels, never to modify inputs.
        if valid_mask is not None:
            mask_flat = valid_mask.flatten()
            n_pixels = len(mask_flat)
            n_total = len(probs)
            mask_expanded = np.tile(mask_flat, n_total // n_pixels + 1)[:n_total]
            valid_idx = mask_expanded > 0
            probs = probs[valid_idx]
            targets = targets[valid_idx]

        valid = ~(np.isnan(probs) | np.isnan(targets))
        probs = probs[valid]
        targets = targets[valid]

        n_pos = (targets == 1).sum()
        print(f"   Total valid pixels: {len(probs):,}")
        print(f"   Positive samples: {n_pos:,} ({100*n_pos/len(targets):.4f}%)")

        thresholds = np.arange(0.01, 0.99, 0.005)
        best_thr, best_metrics = find_best_threshold(
            probs, targets, thresholds, metric=self.optimization_metric
        )

        print(f"\n✅ Optimized threshold ({split_name}): {best_thr:.3f}")
        print(f"   CSI: {best_metrics['csi']:.4f}")
        print(f"   MCC: {best_metrics['mcc']:.4f}")
        print(f"   Precision: {best_metrics['precision']:.4f}")
        print(f"   Recall: {best_metrics['recall']:.4f}")
        print(f"   F1: {best_metrics['f1']:.4f}")

        output_dir = self.paths["grid_search_results"] / "threshold_optimization"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "region": self.config.region,
            "spi_threshold": self.config.spi.threshold,
            "p": p,
            "q": q,
            "split": split_name,
            "best_threshold": float(best_thr),
            "metrics": {
                "csi": float(best_metrics["csi"]),
                "mcc": float(best_metrics["mcc"]),
                "precision": float(best_metrics["precision"]),
                "recall": float(best_metrics["recall"]),
                "f1": float(best_metrics["f1"]),
                "far": float(best_metrics.get("far", 0.0)),
                "bias": float(best_metrics.get("bias", 1.0)),
            },
            "n_samples": len(indices),
            "n_positive_pixels": int(n_pos),
        }

        output_path = output_dir / f"threshold_optimization_p{p}_q{q}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n📁 Result saved to: {output_path}")
        print("=" * 70)

        return best_thr, best_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test-time threshold optimizer")
    parser.add_argument("--region", type=str, default="Sul", help="Region")
    parser.add_argument("--threshold-spi", type=float, default=-2.0, help="SPI threshold")
    parser.add_argument("--no-fallback", action="store_true", help="Do not use validation as a fallback")

    args = parser.parse_args()

    config = ExperimentConfig()
    config.region = args.region
    config.spi.threshold = args.threshold_spi

    from config.paths import get_data_path
    base_data_path = get_data_path()

    optimizer = ThresholdOptimizer(config, base_data_path)
    optimizer.optimize(use_validation_fallback=not args.no_fallback)