"""grid_search.py - Hyperparameter grid search with result tracking."""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from config import ExperimentConfig, get_paths
from data import load_region_timeseries, ClimateNormalizer, ClimateDataset, load_spi_cache
from data.augmentation import get_augmenter
from models import ConvLSTMPredictor
from training import PredictorTrainer
from utils import set_reproducible_seeds
from utils.logger import Logger, Colors


class GridSearch:
    """
    Grid search for hyperparameter optimization.

    Performs systematic search over p (history length) and q (forecast horizon)
    values, evaluating models using the specified optimization metric.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        base_data_path: Path,
        optimization_metric: str = None,
    ):
        self.config = config
        self.base_data_path = base_data_path
        self.threshold = config.spi.threshold
        self.use_transfer_learning = config.use_transfer_learning
        self.logger = Logger()

        self.optimization_metric = (
            optimization_metric or config.optimization.primary_metric
        )
        self.secondary_metric = config.optimization.secondary_metric

        self.suffix = f"thr_{abs(self.threshold):.1f}_tl_{self.use_transfer_learning}"
        self.paths = get_paths(config.region, threshold=self.threshold)

        self.results_dir = self.paths["grid_search_results"] / self.suffix
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None

        set_reproducible_seeds(config.random_seed)

        self.logger.info(f"✅ GridSearch initialized for region: {config.region}")
        self.logger.info(f"   Primary metric: {self.optimization_metric.upper()}")
        self.logger.info(f"   Secondary metric: {self.secondary_metric.upper()}")

    # =========================================================================
    # DISPLAY METHODS
    # =========================================================================

    def _print_header(self, text: str, char: str = "=", width: int = 70) -> None:
        """Print a formatted header."""
        print(f"\n{Colors.BOLD}{Colors.HEADER}{char * width}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.HEADER}{text:^{width}}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.HEADER}{char * width}{Colors.RESET}")

    def _print_section(self, text: str, char: str = "-", width: int = 70) -> None:
        """Print a formatted section title."""
        print(f"\n{Colors.CYAN}{char * width}{Colors.RESET}")
        print(f"{Colors.CYAN}  {text}{Colors.RESET}")
        print(f"{Colors.CYAN}{char * width}{Colors.RESET}")

    def _print_progress(self, current: int, total: int, p: int, q: int) -> None:
        """Print a progress bar for the current combination."""
        bar_len = 30
        percent = current / total
        filled = int(bar_len * percent)
        bar = "█" * filled + "░" * (bar_len - filled)

        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]

        print(f"\n┌{'─' * 68}┐")
        print(f"│ {Colors.BOLD}[{current:>3}/{total}]{Colors.RESET} {bar} {percent:>6.1%} │")
        print(f"│ {'─' * 68} │")
        print(f"│ p={p:>2}  q={q:>2}  ⏱  {elapsed_str} │")
        print(f"└{'─' * 68}┘")

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def load_data(self) -> Dict[str, Any]:
        """Load and preprocess data for grid search."""
        self._print_section("📂 LOADING DATA")

        out = load_region_timeseries(self.base_data_path, self.config)

        data = out["data"]
        years = out["years"]
        months = out["months"]
        valid_mask = out["valid_mask"]

        self._validate_mask(valid_mask, data)

        split_info = self.config.split.get_end_indices_gs()
        time_idx = np.array([y * 12 + (m - 1) for y, m in zip(years, months)])

        train_mask = time_idx <= split_info["train_end"]
        val_mask = (time_idx > split_info["train_end"]) & (time_idx <= split_info["val_end"])

        data_train = data[train_mask]
        data_val = data[val_mask]
        months_train = months[train_mask]
        months_val = months[val_mask]

        train_period = (
            f"{years[train_mask][0]}-{months[train_mask][0]:02d} "
            f"to {years[train_mask][-1]}-{months[train_mask][-1]:02d}"
        )
        val_period = (
            f"{years[val_mask][0]}-{months[val_mask][0]:02d} "
            f"to {years[val_mask][-1]}-{months[val_mask][-1]:02d}"
        )

        print(f"  {Colors.GREEN}Training{Colors.RESET}    : {len(data_train):>4} months  ({train_period})")
        print(f"  {Colors.CYAN}Validation{Colors.RESET}  : {len(data_val):>4} months  ({val_period})")

        # Normalize data
        normalizer = ClimateNormalizer(self.config.data.bands)
        data_train_norm = normalizer.fit_transform(data_train, months_train, valid_mask)
        data_val_norm = normalizer.transform(data_val, months_val, valid_mask)

        # Save normalizer
        normalizer_path = self.paths["autoencoder_dir"] / "normalizer.json"
        normalizer_path.parent.mkdir(parents=True, exist_ok=True)
        normalizer.save(normalizer_path)
        print(f"  ✅ Normalizer saved to: {normalizer_path}")

        backup_path = self.paths["grid_search_dir"] / "normalizer.json"
        normalizer.save(backup_path)

        # Load SPI
        spi, _ = load_spi_cache(self.config.spi.scale, self.paths["spi_cache_dir"])
        spi_train = spi[train_mask] if spi is not None else None
        spi_val = spi[val_mask] if spi is not None else None

        return {
            "data_train": np.nan_to_num(data_train_norm, nan=0.0),
            "data_val": np.nan_to_num(data_val_norm, nan=0.0),
            "months_train": months_train,
            "months_val": months_val,
            "spi_train": spi_train,
            "spi_val": spi_val,
            "valid_mask": valid_mask,
        }

    def _validate_mask(self, valid_mask: np.ndarray, data: np.ndarray) -> None:
        """Validate the validity mask and log statistics."""
        if valid_mask is None:
            self.logger.warning("⚠️  No validity mask provided")
            return

        total_pixels = valid_mask.size
        valid_pixels = valid_mask.sum()
        invalid_pixels = total_pixels - valid_pixels

        print(f"\n  📊 Validity Mask:")
        print(f"     Total pixels: {total_pixels:,}")
        print(f"     Valid: {valid_pixels:,} ({valid_pixels/total_pixels:.1%})")
        print(f"     Invalid: {invalid_pixels:,} ({invalid_pixels/total_pixels:.1%})")

        if data is not None and len(data) > 0:
            zero_pixels = (data[0] == 0).all(axis=-1)
            zero_valid = zero_pixels & valid_mask

            print(f"\n  📊 Zero-value (non-drought) pixels:")
            print(f"     Total: {zero_pixels.sum():,}")
            print(f"     Valid in mask: {zero_valid.sum():,}")

            if zero_valid.sum() == 0 and zero_pixels.sum() > 0:
                self.logger.warning("\n  ⚠️  WARNING: No non-drought pixels are considered valid!")
                self.logger.warning("     The model will NEVER see non-drought examples during training.")
            else:
                print(f"     ✅ {zero_valid.sum():,} non-drought pixels are valid")

        if data is not None:
            nan_pixels = np.isnan(data).any(axis=-1).any(axis=0)
            nan_invalid = nan_pixels & ~valid_mask
            if nan_invalid.sum() > 0:
                print(f"\n  ℹ️  NaN/Inf pixels marked as invalid: {nan_invalid.sum():,}")

    def load_autoencoder(self) -> Optional[Dict[str, Any]]:
        """Load pretrained autoencoder checkpoint for transfer learning."""
        autoencoder_path = self.paths["autoencoder_dir"] / "model.pth"
        if not autoencoder_path.exists():
            return None

        checkpoint = torch.load(
            autoencoder_path,
            map_location=self.config.device,
            weights_only=False
        )

        # Add metadata about anomalies
        if hasattr(self.config.autoencoder, 'use_anomalies'):
            checkpoint["use_anomalies"] = self.config.autoencoder.use_anomalies

        return checkpoint

    # =========================================================================
    # AUGMENTATION SETUP
    # =========================================================================

    def _create_augmenter(self) -> Optional[object]:
        """Create data augmenter if enabled."""
        if not self.config.augmentation.enabled:
            return None

        augmenter = get_augmenter(
            augment_type=self.config.augmentation.augment_type,
            severity_factor=self.config.augmentation.severity_factor,
            expansion_factor=self.config.augmentation.expansion_factor,
            prob=self.config.augmentation.prob,
            seed=self.config.random_seed,
        )

        aug_info = augmenter.get_info() if hasattr(augmenter, 'get_info') else {}
        self.logger.info(f"  🔄 Augmentation enabled:")
        self.logger.info(f"     Type: {self.config.augmentation.augment_type}")
        self.logger.info(f"     Severity factor: {aug_info.get('severity_factor', 0.3)}")
        self.logger.info(f"     Expansion factor: {aug_info.get('expansion_factor', 0.2)}")
        self.logger.info(f"     Probability: {aug_info.get('prob', 0.5)}")

        return augmenter

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    def run(self) -> None:
        """Run the grid search."""
        self.start_time = datetime.now()

        self._print_header("🔍 GRID SEARCH")

        # Display experiment configuration
        self._display_experiment_config()

        # Load data
        data_dict = self.load_data()
        augmenter = self._create_augmenter()

        # Load autoencoder for transfer learning
        ae_checkpoint = self.load_autoencoder() if self.use_transfer_learning else None
        self._display_transfer_learning_status(ae_checkpoint)

        # Run grid search
        total_combinations = len(self.config.p_values) * len(self.config.q_values)
        current = 0

        print(f"\n{'─' * 70}")
        print(f"  🚀 Starting grid search with {total_combinations} combinations")
        print(f"{'─' * 70}")

        for p in self.config.p_values:
            for q in self.config.q_values:
                current += 1

                # Validate data sufficiency
                if not self._validate_data_sufficiency(p, q, data_dict):
                    continue

                self._print_progress(current, total_combinations, p, q)

                # Create datasets
                train_ds, val_ds = self._create_datasets(p, q, data_dict, augmenter)

                if len(train_ds) == 0 or len(val_ds) == 0:
                    print("  ⚠️  Dataset empty! Skipping...")
                    continue

                # Create and train model
                result = self._train_and_evaluate(p, q, train_ds, val_ds, ae_checkpoint)

                if result is not None:
                    self.results.append(result)

                # Cleanup
                del train_ds, val_ds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Save results
        self.save_results()

        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]
        self._print_header(f"✅ GRID SEARCH COMPLETED  ⏱  {elapsed_str}")

    # =========================================================================
    # HELPER METHODS FOR RUN
    # =========================================================================

    def _display_experiment_config(self) -> None:
        """Display experiment configuration."""
        ds_info = self.config.get_downsample_info(self.config.region)
        ds_h, ds_w = self.config.get_downsample(self.config.region)

        print(f"  {Colors.BOLD}Region{Colors.RESET}           : {self.config.region}")
        print(f"  {Colors.BOLD}Downsampling{Colors.RESET}     : {ds_h}x{ds_w}")
        print(f"  {Colors.BOLD}Preservation{Colors.RESET}     : {ds_info['preservation_estimate']}")
        print(f"  {Colors.BOLD}Area Reduction{Colors.RESET}   : {ds_info['area_reduction']}x")
        print(f"  {Colors.BOLD}Threshold{Colors.RESET}        : {self.threshold}")
        print(f"  {Colors.BOLD}Transfer Learning{Colors.RESET}: {self.use_transfer_learning}")
        print(f"  {Colors.BOLD}Optimization{Colors.RESET}     : {self.optimization_metric.upper()}")
        print(f"  {Colors.BOLD}P values{Colors.RESET}         : {self.config.p_values}")
        print(f"  {Colors.BOLD}Q values{Colors.RESET}         : {self.config.q_values}")
        print(f"  {Colors.BOLD}Augmentation{Colors.RESET}     : {self.config.augmentation.enabled}")
        print(f"  {Colors.BOLD}Suffix{Colors.RESET}           : {self.suffix}")
        print(f"{'─' * 68}")

    def _display_transfer_learning_status(self, ae_checkpoint: Optional[Dict]) -> None:
        """Display transfer learning status."""
        if ae_checkpoint:
            print("\n  🧠 Using autoencoder for transfer learning")
            print("  📌 Apenas o encoder será transferido (attention aleatória)")
        else:
            print("\n  🔧 Training models from scratch")

    def _validate_data_sufficiency(self, p: int, q: int, data_dict: Dict) -> bool:
        """Validate if there is enough data for the given p and q."""
        max_samples = len(data_dict["data_val"])
        min_required = p + q

        if min_required >= max_samples:
            print(f"\n  ⏭️  Skipping p={p}, q={q} (p+q={min_required} >= val_size={max_samples})")
            return False

        train_max = len(data_dict["data_train"])
        if min_required >= train_max:
            print(f"\n  ⏭️  Skipping p={p}, q={q} (p+q={min_required} >= train_size={train_max})")
            return False

        return True

    def _create_datasets(
        self,
        p: int,
        q: int,
        data_dict: Dict,
        augmenter: Optional[object]
    ) -> tuple:
        """Create training and validation datasets."""
        train_ds = ClimateDataset(
            data=data_dict["data_train"],
            spi=data_dict["spi_train"],
            months=data_dict["months_train"],
            p=p, q=q,
            spi_threshold=self.config.spi.threshold,
            valid_mask=data_dict["valid_mask"],
            use_weighted_sampling=self.config.imbalance.use_weighted_sampling,
            temporal_decay=True,
            mode="classification",
            augmenter=augmenter,
            seed=self.config.random_seed,
        )

        val_ds = ClimateDataset(
            data=data_dict["data_val"],
            spi=data_dict["spi_val"],
            months=data_dict["months_val"],
            p=p, q=q,
            spi_threshold=self.config.spi.threshold,
            valid_mask=data_dict["valid_mask"],
            use_weighted_sampling=False,
            temporal_decay=True,
            mode="classification",
            augmenter=None,
            seed=self.config.random_seed,
        )

        print(f"  📊 Training samples: {len(train_ds):>4}")
        print(f"  📊 Validation samples: {len(val_ds):>4}")

        if augmenter is not None:
            aug_stats = train_ds.get_augmentation_stats()
            print(f"  🔄 Augmentation: {aug_stats}")

        return train_ds, val_ds

    def _train_and_evaluate(
        self,
        p: int,
        q: int,
        train_ds: ClimateDataset,
        val_ds: ClimateDataset,
        ae_checkpoint: Optional[Dict]
    ) -> Optional[Dict]:
        """Train and evaluate a model for given parameters."""
        # Create model
        model = ConvLSTMPredictor(
            self.config.get_model_config("predictor")
        ).to(self.config.device)

        # ================================================================
        # TRANSFER LEARNING: ENCODER + ATTENTION
        # ================================================================
        if ae_checkpoint is not None:
            try:
                model.load_encoder_from_autoencoder(
                    ae_checkpoint,
                    load_attention=True  
                )
                print(f"  🧠 Encoder: transferred ✅")
                print(f"  🧠 Attention: transferred ✅")
            except Exception as e:
                print(f"  ⚠️  Error transferring encoder/attention: {e}")
                print(f"  🔧 Training from scratch...")
                ae_checkpoint = None

        # Set save path
        model_dir = (
            self.paths["grid_search_pretrained"]
            if ae_checkpoint
            else self.paths["grid_search_scratch"]
        )
        save_path = model_dir / f"model_p{p}_q{q}.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Create trainer
        trainer = PredictorTrainer(
            model,
            train_ds,
            val_ds,
            self.config,
            save_path,
            freeze_encoder_first_epoch=bool(ae_checkpoint),
            optimization_metric=self.optimization_metric,
            calibrate=self.config.training.calibrate,
        )

        # Train
        try:
            best_score = trainer.train()
        except Exception as e:
            print(f"  ❌ Training error: {e}")
            return None

        # Load checkpoint
        try:
            ckpt = torch.load(save_path, map_location=self.config.device, weights_only=False)
        except Exception as e:
            print(f"  ⚠️  Error loading checkpoint: {e}")
            ckpt = {}

        # Build result
        result = {
            "region": self.config.region,
            "p": p,
            "q": q,
            "transfer_learning": self.use_transfer_learning,
            "load_attention": True if ae_checkpoint is not None else False,
            f"best_{self.optimization_metric}": ckpt.get(
                "best_score", best_score if best_score is not None else 0.0
            ),
            "best_csi": ckpt.get("best_csi", 0.0),
            "best_mcc": ckpt.get("best_mcc", 0.0),
            "threshold": ckpt.get("best_threshold", 0.14),
            "precision": ckpt.get("precision", 0.0),
            "recall": ckpt.get("recall", 0.0),
            "f1": ckpt.get("f1", 0.0),
            "epoch": ckpt.get("epoch", 0),
            "augmentation": self.config.augmentation.enabled,
        }

        if "final_metrics" in ckpt:
            final_metrics = ckpt["final_metrics"]
            result["final_csi"] = final_metrics.get("csi", 0.0)
            result["final_mcc"] = final_metrics.get("mcc", 0.0)
            result["final_threshold"] = ckpt.get("best_threshold", 0.14)

        # Cleanup
        del model, trainer

        return result

    # =========================================================================
    # RESULTS SAVING
    # =========================================================================

    def save_results(self) -> None:
        """Save grid search results to files."""
        if not self.results:
            print("  ⚠️  No results to save")
            return

        self.results_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.results)
        score_col = f"best_{self.optimization_metric}"
        df = df.sort_values(score_col, ascending=False)

        # Save to Excel
        excel_path = self.results_dir / f"grid_search_results_{self.suffix}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="All Results", index=False)

            stats_df = df.groupby(["p", "q"]).agg({
                score_col: ["mean", "std", "max", "count"],
                "best_csi": "mean",
                "best_mcc": "mean",
                "threshold": "mean"
            }).round(4)
            stats_df.to_excel(writer, sheet_name="Stats by p,q")

            best_by_pq = df.loc[df.groupby(["p", "q"])[score_col].idxmax()]
            best_by_pq[
                ["p", "q", score_col, "best_csi", "best_mcc",
                 "threshold", "precision", "recall", "f1", "augmentation"]
            ].to_excel(writer, sheet_name="Best by p,q", index=False)

        print(f"\n  📊 Results saved to: {excel_path}")

        self.save_best_configuration()

    def save_best_configuration(self) -> None:
        """Save the best configuration and display results."""
        if not self.results:
            print("  ⚠️  No results to save")
            return

        df = pd.DataFrame(self.results)

        primary_col = f"best_{self.optimization_metric}"
        secondary_col = f"best_{self.secondary_metric}"

        # Sort by primary and secondary metrics
        if primary_col in df.columns and secondary_col in df.columns:
            df_sorted = df.sort_values([primary_col, secondary_col], ascending=[False, False])
        else:
            df_sorted = df.sort_values(primary_col, ascending=False)

        best = df_sorted.iloc[0].to_dict()
        best = self._convert_numpy_types(best)

        top5_df = df_sorted.head(5)[
            ["p", "q", "transfer_learning", primary_col, secondary_col,
             "best_csi", "best_mcc", "threshold", "precision", "recall", "f1", "augmentation"]
        ]
        top5 = self._convert_list_types(top5_df.to_dict("records"))

        stats = {
            "total_combinations": len(self.results),
            "optimization_metric": self.optimization_metric,
            "secondary_metric": self.secondary_metric,
            f"best_{self.optimization_metric}": float(best[primary_col]),
            f"best_{self.secondary_metric}": float(best[secondary_col]) if secondary_col in best else 0.0,
            "best_csi": float(best["best_csi"]),
            "best_mcc": float(best["best_mcc"]),
            "best_p": int(best["p"]),
            "best_q": int(best["q"]),
            "transfer_learning": bool(best.get("transfer_learning", self.use_transfer_learning)),
            "augmentation_used": bool(best.get("augmentation", False)),
            "mean_score": float(df[primary_col].mean()),
            "std_score": float(df[primary_col].std()),
            "median_score": float(df[primary_col].median()),
        }

        # Save JSON summary
        config_summary = {
            "experiment_info": {
                "region": self.config.region,
                "spi_scale": self.config.spi.scale,
                "spi_threshold": self.config.spi.threshold,
                "transfer_learning_used": self.use_transfer_learning,
                "optimization_metric": self.optimization_metric,
                "secondary_metric": self.secondary_metric,
                "use_weighted_sampling": self.config.imbalance.use_weighted_sampling,
                "augmentation_enabled": self.config.augmentation.enabled,
                "augmentation_type": self.config.augmentation.augment_type,
            },
            "best_configuration": best,
            "top5_configurations": top5,
            "statistics": stats,
        }

        output_path = self.results_dir / f"best_configuration_{self.suffix}.json"
        with open(output_path, "w") as f:
            json.dump(config_summary, f, indent=2)

        # Display results
        self._display_best_configuration(best, primary_col, secondary_col)
        self._display_top5(top5, primary_col, secondary_col)

        # Save summary report
        self._save_summary_report(best, top5, primary_col, secondary_col)

    def _convert_numpy_types(self, obj: Dict) -> Dict:
        """Convert numpy types to Python types."""
        for key, value in obj.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                obj[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                obj[key] = float(value)
            elif isinstance(value, np.bool_):
                obj[key] = bool(value)
        return obj

    def _convert_list_types(self, records: List[Dict]) -> List[Dict]:
        """Convert numpy types in a list of records."""
        return [self._convert_numpy_types(record) for record in records]

    def _display_best_configuration(self, best: Dict, primary_col: str, secondary_col: str) -> None:
        """Display the best configuration."""
        self._print_header(f"🏆 BEST CONFIGURATION ({self.optimization_metric.upper()})")

        print(f"  {'Parameter':<20} {'Value':>15}")
        print(f"  {'─' * 38}")
        print(f"  {'p':<20} {best['p']:>15}")
        print(f"  {'q':<20} {best['q']:>15}")
        print(f"  {self.optimization_metric.upper():<20} {best[primary_col]:>15.4f}")
        print(f"  {self.secondary_metric.upper():<20} {best[secondary_col]:>15.4f}")
        print(f"  {'CSI':<20} {best['best_csi']:>15.4f}")
        print(f"  {'MCC':<20} {best['best_mcc']:>15.4f}")
        print(f"  {'Threshold':<20} {best['threshold']:>15.3f}")
        print(f"  {'Precision':<20} {best['precision']:>15.4f}")
        print(f"  {'Recall':<20} {best['recall']:>15.4f}")
        print(f"  {'F1':<20} {best['f1']:>15.4f}")
        print(f"  {'Transfer Learning':<20} {str(best['transfer_learning']):>15}")
        print(f"  {'Augmentation':<20} {str(best.get('augmentation', False)):>15}")

        if "final_csi" in best:
            print(f"  {'Final CSI':<20} {best['final_csi']:>15.4f}")
        if "final_mcc" in best:
            print(f"  {'Final MCC':<20} {best['final_mcc']:>15.4f}")

        print(f"{'─' * 38}")

    def _display_top5(self, top5: List[Dict], primary_col: str, secondary_col: str) -> None:
        """Display the top 5 configurations."""
        self._print_section(f"📊 TOP 5 ({self.optimization_metric.upper()})")

        print(f"  {'#':<3} {'p':<4} {'q':<4} {self.optimization_metric.upper():<10} "
              f"{self.secondary_metric.upper():<10} {'CSI':<10} {'MCC':<10} "
              f"{'Threshold':<12} {'Aug':<5}")
        print(f"  {'─' * 75}")

        for i, cfg in enumerate(top5, 1):
            aug_str = "✅" if cfg.get('augmentation', False) else "❌"
            print(
                f"  {i:<3} {cfg['p']:<4} {cfg['q']:<4} "
                f"{cfg[primary_col]:<10.4f} {cfg[secondary_col]:<10.4f} "
                f"{cfg['best_csi']:<10.4f} {cfg['best_mcc']:<10.4f} "
                f"{cfg['threshold']:<12.3f} {aug_str:<5}"
            )
        print(f"  {'─' * 75}")

    def _save_summary_report(self, best: Dict, top5: List[Dict], primary_col: str, secondary_col: str) -> None:
        """Save a text summary report."""
        report_path = self.results_dir / f"grid_search_summary_{self.suffix}.txt"

        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("GRID SEARCH SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Region: {self.config.region}\n")
            f.write(f"SPI Threshold: {self.threshold}\n")
            f.write(f"Transfer Learning: {self.use_transfer_learning}\n")
            f.write(f"Optimization Metric (Primary): {self.optimization_metric.upper()}\n")
            f.write(f"Optimization Metric (Secondary): {self.secondary_metric.upper()}\n")
            f.write(f"Augmentation: {self.config.augmentation.enabled}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("BEST CONFIGURATION:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  p: {best['p']}\n")
            f.write(f"  q: {best['q']}\n")
            f.write(f"  {self.optimization_metric.upper()}: {best[primary_col]:.4f}\n")
            f.write(f"  {self.secondary_metric.upper()}: {best[secondary_col]:.4f}\n")
            f.write(f"  CSI: {best['best_csi']:.4f}\n")
            f.write(f"  MCC: {best['best_mcc']:.4f}\n")
            f.write(f"  Threshold: {best['threshold']:.3f}\n")
            f.write(f"  Precision: {best['precision']:.4f}\n")
            f.write(f"  Recall: {best['recall']:.4f}\n")
            f.write(f"  F1: {best['f1']:.4f}\n")
            f.write(f"  Augmentation: {best.get('augmentation', False)}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("TOP 5 CONFIGURATIONS:\n")
            f.write("-" * 60 + "\n")
            for i, cfg in enumerate(top5, 1):
                f.write(
                    f"  {i}. p={cfg['p']}, q={cfg['q']}, "
                    f"{self.optimization_metric.upper()}={cfg[primary_col]:.4f}, "
                    f"{self.secondary_metric.upper()}={cfg[secondary_col]:.4f}, "
                    f"CSI={cfg['best_csi']:.4f}, MCC={cfg['best_mcc']:.4f}, "
                    f"Aug={cfg.get('augmentation', False)}\n"
                )

        print(f"\n  📄 Summary report saved to: {report_path}")