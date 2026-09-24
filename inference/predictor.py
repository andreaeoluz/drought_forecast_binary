"""Inference predictor for drought forecasting."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json

from config import ExperimentConfig
from config.paths import get_paths
from data import load_region_timeseries, ClimateNormalizer, load_spi_cache
from models import ConvLSTMPredictor
from evaluation.metrics import find_best_threshold
from utils import set_reproducible_seeds
from utils.logger import Logger
from utils.spatial import postprocess_binary_mask


class InferencePredictor:
    """Handles inference on the test period with optional calibration and fallback."""

    def __init__(
        self,
        config: ExperimentConfig,
        base_data_path: Path,
        p: int,
        q: int,
        model_type: str = "pretrained",
        optimization_metric: str = None,
        use_calibration: bool = None,
        use_validation_fallback: bool = True,
        fixed_threshold: Optional[float] = None,
        model_dir: Optional[Path] = None,
    ):
        """
        Initialize the inference predictor.

        Args:
            config: Experiment configuration.
            base_data_path: Path to the raw data directory.
            p: History length.
            q: Forecast horizon.
            model_type: 'pretrained' or 'scratch'.
            optimization_metric: Metric for threshold optimization ('mcc' or
                'csi'). If None, uses config.optimization.primary_metric.
            use_calibration: Whether to apply probability calibration. If
                None, uses config.training.calibrate.
            use_validation_fallback: Whether to use validation data for
                threshold calibration.
            fixed_threshold: If provided, use this threshold instead of
                optimizing one.
            model_dir: Optional custom directory to search for the model
                checkpoint (and calibrator.pkl) before the standard
                grid-search locations.
        """
        self.config = config
        self.base_data_path = base_data_path
        self.p = p
        self.q = q
        self.model_type = model_type
        self.model_dir = Path(model_dir) if model_dir else None

        if optimization_metric is None:
            self.optimization_metric = config.optimization.primary_metric
        else:
            self.optimization_metric = optimization_metric

        if use_calibration is None:
            self.use_calibration = getattr(config.training, 'calibrate', False)
        else:
            self.use_calibration = use_calibration

        self.use_validation_fallback = use_validation_fallback
        self.fixed_threshold = fixed_threshold
        self.calibrator = None
        self.logger = Logger()

        self.paths = get_paths(
            config.region,
            threshold=config.spi.threshold,
            p=p,
            q=q,
            model_type=model_type,
        )

        set_reproducible_seeds(config.random_seed)

        self._load_data()
        self._load_model()
        self._load_calibrator()
        self._setup_postprocessing()

        self.logger.info("📊 InferencePredictor initialized:")
        self.logger.info(f"   Optimization metric: {self.optimization_metric.upper()}")
        self.logger.info(f"   Calibration: {'✅ ENABLED' if self.use_calibration else '❌ DISABLED'}")
        self.logger.info(f"   Validation fallback: {'✅ ENABLED' if self.use_validation_fallback else '❌ DISABLED'}")
        if self.fixed_threshold is not None:
            self.logger.info(f"   Fixed threshold: {self.fixed_threshold:.3f}")

    def _setup_postprocessing(self):
        """Set up spatial post-processing parameters (min object/hole area)."""
        min_area_original = getattr(self.config, 'min_area_original_pixels', None)
        if min_area_original is not None:
            factor_h, _factor_w = self.config.get_downsample(self.config.region)
            min_area = max(1, min_area_original // (factor_h ** 2))
        else:
            min_area = self.config.min_area_downsampled
        self.min_area = min_area
        self.hole_area = max(1, self.min_area // 2)

    def _load_data(self):
        """Load climate data, SPI, and the normalizer."""
        self.logger.info("Loading data...")
        out = load_region_timeseries(self.base_data_path, self.config)
        self.data = out["data"]
        self.years = out["years"]
        self.months = out["months"]
        self.metadata = out["metadata"]
        self.valid_mask = out["valid_mask"]
        self._resize_valid_mask()
        self._load_spi()
        self._load_normalizer()
        self._setup_temporal_indices()

    def _resize_valid_mask(self):
        """Resize the validity mask to match the data resolution."""
        if self.valid_mask is None:
            return
        target_shape = self.data.shape[1:3]
        if self.valid_mask.shape != target_shape:
            from skimage.transform import resize
            self.valid_mask = resize(
                self.valid_mask.astype(np.float32),
                target_shape,
                order=0,
                preserve_range=True
            ).astype(bool)

    def _load_spi(self):
        """Load and resize the cached SPI series."""
        spi, _ = load_spi_cache(self.config.spi.scale, self.paths["spi_cache_dir"])
        self.spi = spi
        if self.spi is None:
            return
        target_shape = self.data.shape[1:3]
        if self.spi.shape[1:] != target_shape:
            from skimage.transform import resize
            spi_resized = np.zeros(
                (self.spi.shape[0], target_shape[0], target_shape[1]),
                dtype=self.spi.dtype
            )
            for t in range(self.spi.shape[0]):
                spi_resized[t] = resize(
                    self.spi[t],
                    target_shape,
                    order=0,
                    preserve_range=True
                )
            self.spi = spi_resized

    def _load_normalizer(self):
        """Load the climate normalizer matching this model's training.

        A pretrained/transfer-learning model's encoder was fine-tuned on
        data normalized with the autoencoder's own normalizer (see
        GridSearch.load_data) - it must be served with that exact same
        normalizer. A scratch model was trained on a fresh normalizer fit
        on the classification task's own train split, saved separately so
        the two never collide (they legitimately differ).
        """
        if self.model_type == "scratch":
            candidates = [
                self.paths["grid_search_dir"] / "normalizer_scratch.json",
                self.paths["autoencoder_dir"] / "normalizer.json",  # legacy fallback
            ]
        else:
            candidates = [
                self.paths["autoencoder_dir"] / "normalizer.json",
                self.paths["grid_search_dir"] / "normalizer_pretrained.json",
            ]

        normalizer_path = next((p for p in candidates if p.exists()), None)
        if normalizer_path is None:
            raise FileNotFoundError(f"Normalizer not found among: {candidates}")

        self.normalizer = ClimateNormalizer.load(normalizer_path)
        self.logger.success(f"Normalizer loaded ({self.model_type}): {normalizer_path}")

    def _setup_temporal_indices(self):
        """Set up temporal indices for the test and validation periods."""
        self.time_idx = np.array([y * 12 + (m - 1) for y, m in zip(self.years, self.months)])
        split = self.config.split

        test_start = split.ym_to_int(split.test[0])
        test_end = split.ym_to_int(split.test[1])
        self.test_mask = (self.time_idx >= test_start) & (self.time_idx <= test_end)
        self.test_indices = np.where(self.test_mask)[0]
        self.test_data = self.data[self.test_mask]
        self.test_months = self.months[self.test_mask]
        self.spi_test = self.spi[self.test_mask] if self.spi is not None else None

        val_start = split.ym_to_int(split.val_gs[0])
        val_end = split.ym_to_int(split.val_gs[1])
        self.val_mask = (self.time_idx >= val_start) & (self.time_idx <= val_end)
        self.val_indices = np.where(self.val_mask)[0]
        self.val_data = self.data[self.val_mask]
        self.val_months = self.months[self.val_mask]
        self.spi_val = self.spi[self.val_mask] if self.spi is not None else None

        self.logger.info(f"Test period: {len(self.test_indices)} months")
        self.logger.info(f"Validation period: {len(self.val_indices)} months")

    def _find_model_path(self) -> Path:
        """Find the model checkpoint path.

        Only searches locations consistent with self.model_type. The
        previous version always probed grid_search_pretrained first
        regardless of model_type, so any "scratch" inference request
        silently loaded the pretrained checkpoint whenever one existed for
        the same (p, q) - which it almost always does - making every
        scratch inference run byte-identical to the pretrained one.
        """
        model_filename = f"model_p{self.p}_q{self.q}.pth"
        type_dir = (
            self.paths["grid_search_pretrained"] if self.model_type == "pretrained"
            else self.paths["grid_search_scratch"]
        )
        candidates = []
        if self.model_dir is not None:
            candidates.append(self.model_dir / model_filename)
        candidates += [
            type_dir / model_filename,
            self.paths["grid_search_dir"] / self.model_type / model_filename,
        ]
        for path in candidates:
            if path.exists():
                self.logger.info(f"Model found: {path}")
                return path
        raise FileNotFoundError(
            f"Model not found for p={self.p}, q={self.q}, type={self.model_type} "
            f"(checked: {[str(c) for c in candidates]})"
        )

    def _load_model(self):
        """Load the trained model checkpoint.

        The model architecture is built to match what's ACTUALLY in the
        checkpoint's state_dict, not blindly from `config.get_model_config`.
        This matters since the addition of the multiscale temporal module
        and the residual adapter (see models/predictor.py): both are
        randomly initialized when the model is constructed, and if a
        checkpoint trained *before* those components existed were loaded
        into a model built with them enabled (as a naive
        `ConvLSTMPredictor(config.get_model_config(...))` would do), the
        random multiscale context would get added into the latent (with
        weight `multiscale_weight`) at inference time even though the model
        never learned to expect it - silently corrupting predictions from
        every pre-upgrade checkpoint instead of failing loudly. Detecting
        the architecture from the checkpoint itself keeps old and new
        checkpoints both loading correctly, regardless of the *current*
        default config.
        """
        checkpoint_path = self._find_model_path()
        self.logger.info(f"Loading model: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]

        model_config = self.config.get_model_config("predictor")
        self._reconcile_architecture_with_checkpoint(model_config, state_dict)

        self.model = ConvLSTMPredictor(model_config).to(self.config.device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            self.logger.debug(f"  (state_dict missing keys, kept at init: {missing})")
        if unexpected:
            self.logger.warning(f"  ⚠️ state_dict had unexpected keys, ignored: {unexpected}")

        self.model.eval()
        self.best_threshold = checkpoint.get("best_threshold", 0.20)
        self.best_csi = checkpoint.get("best_csi", 0.0)
        self.best_mcc = checkpoint.get("best_mcc", 0.0)
        self.logger.success(f"Model loaded (CSI: {self.best_csi:.4f}, MCC: {self.best_mcc:.4f})")

    def _reconcile_architecture_with_checkpoint(self, model_config: dict, state_dict: dict) -> None:
        """
        Mutate `model_config` in place so `use_attention`, `use_multiscale`
        and `use_residual_adapter` match what the checkpoint was actually
        trained with, overriding whatever the current experiment config
        says. Logs a warning whenever an override happens, since it means
        this checkpoint predates (or otherwise differs from) the currently
        configured architecture.
        """
        component_prefixes = {
            "use_attention": "attention.",
            "use_multiscale": "multiscale.",
            "use_residual_adapter": "residual_adapter.",
        }

        for config_key, prefix in component_prefixes.items():
            present_in_checkpoint = any(k.startswith(prefix) for k in state_dict)
            configured = model_config.get(config_key, True)

            if present_in_checkpoint != configured:
                self.logger.warning(
                    f"  ⚠️ Architecture mismatch for '{config_key}': config={configured}, "
                    f"checkpoint has it={present_in_checkpoint}. Using the checkpoint's "
                    f"architecture (this model was likely trained before/after this "
                    f"component was added)."
                )
                model_config[config_key] = present_in_checkpoint

    def _load_calibrator(self):
        """Load the probability calibrator, if enabled."""
        if not self.use_calibration:
            return
        self.logger.info("Loading calibrator...")
        checkpoint_path = self._find_model_path()

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
            if "calibrator" in checkpoint and checkpoint["calibrator"] is not None:
                self.calibrator = checkpoint["calibrator"]
                self.logger.success("Calibrator loaded from checkpoint")
                return
        except Exception as e:
            self.logger.debug(f"Could not load calibrator from checkpoint: {e}")

        calibrator_paths = []
        if self.model_dir is not None:
            calibrator_paths.append(self.model_dir / "calibrator.pkl")
        calibrator_paths += [
            checkpoint_path.parent / "calibrator.pkl",
            self.paths["grid_search_dir"] / "calibrator.pkl",
        ]
        for path in calibrator_paths:
            if path.exists():
                try:
                    import joblib
                    self.calibrator = joblib.load(path)
                    self.logger.success(f"Calibrator loaded: {path}")
                    return
                except Exception as e:
                    self.logger.debug(f"Could not load calibrator from {path}: {e}")

        self.logger.warning("Calibrator not found. Using raw probabilities.")

    def _calibrate_probs(self, probs: np.ndarray) -> np.ndarray:
        """Apply the fitted calibrator to raw probabilities."""
        if self.calibrator is None or not self.use_calibration:
            return probs

        original_shape = probs.shape
        probs_flat = probs.flatten()

        calibrated = self.calibrator.transform(probs_flat)
        return calibrated.reshape(original_shape)

    def _prepare_data(self, data_subset: np.ndarray, months_subset: np.ndarray) -> np.ndarray:
        """Prepare data for inference (normalize + channels-first)."""
        data_norm = self.normalizer.transform(data_subset, months_subset, self.valid_mask)
        data_norm = np.nan_to_num(data_norm, nan=0.0)
        return np.transpose(data_norm, (0, 3, 1, 2))

    def _get_sample_indices(self, data_len: int) -> List[int]:
        """Get valid sample indices for a data subset."""
        min_required = self.p + self.q
        if data_len <= min_required:
            return []
        # target_idx = t + q - 1 (see predict_subset), so the last usable
        # t is data_len - q, not data_len - q - 1.
        return list(range(self.p, data_len - self.q + 1))

    def _filter_valid_pixels(self, probs_stack: np.ndarray, targets_stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter to valid pixels using the validity mask."""
        probs_flat = probs_stack.flatten()
        targets_flat = targets_stack.flatten()

        if self.valid_mask is not None:
            mask_flat = self.valid_mask.flatten()
            n_pixels = len(mask_flat)
            n_total = len(probs_flat)
            mask_expanded = np.tile(mask_flat, n_total // n_pixels + 1)[:n_total]
            valid_idx = mask_expanded > 0
            probs_flat = probs_flat[valid_idx]
            targets_flat = targets_flat[valid_idx]

        valid = ~(np.isnan(probs_flat) | np.isnan(targets_flat))
        return probs_flat[valid], targets_flat[valid]

    def _optimize_threshold(self, probs: np.ndarray, targets: np.ndarray) -> Tuple[float, Dict]:
        """Optimize the decision threshold over an adaptive range."""
        thresholds = self.config.calibration_thresholds

        p1 = np.percentile(probs, 1)
        p99 = np.percentile(probs, 99)

        thresholds = [t for t in thresholds if p1 <= t <= p99]

        if len(thresholds) < 10:
            low = max(0.001, p1 - 0.02)
            high = min(0.999, p99 + 0.02)
            thresholds = np.arange(low, high + 0.005, 0.005)

        return find_best_threshold(probs, targets, thresholds, metric=self.optimization_metric)

    def predict_subset(
        self,
        data_subset: np.ndarray,
        months_subset: np.ndarray,
        spi_subset: np.ndarray,
        fixed_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Make predictions on a data subset.

        Args:
            data_subset: Climate data (T, H, W, C).
            months_subset: Month indices (T,).
            spi_subset: SPI values (T, H, W).
            fixed_threshold: If provided, use this threshold instead of optimizing.

        Returns:
            Dict with predictions, targets, metrics, and threshold.
        """
        data_ch = self._prepare_data(data_subset, months_subset)
        T, C, H, W = data_ch.shape
        indices = self._get_sample_indices(T)

        if not indices:
            return {"probs": None, "targets": None, "metrics": None, "n_samples": 0}

        all_probs = []
        all_targets = []

        binary_mask = (spi_subset <= self.config.spi.threshold).astype(np.float32)

        mask_tensor = None
        if self.valid_mask is not None:
            mask_tensor = torch.from_numpy(self.valid_mask.astype(np.float32)).to(self.config.device)

        with torch.no_grad():
            for t in indices:
                # Matches ClimateDataset's convention: window is
                # data_ch[t-p:t] (ends at t-1), target is the qth month
                # after that window, i.e. B_{(t-p)+p+q-1} = B_{t+q-1}.
                target_idx = t + self.q - 1
                x_seq = data_ch[t - self.p:t]
                x_tensor = torch.from_numpy(x_seq).float().unsqueeze(0).to(self.config.device)
                logits, _ = self.model(x_tensor, mask=mask_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
                all_probs.append(probs)
                all_targets.append(binary_mask[target_idx])

        probs_stack = np.stack(all_probs)
        targets_stack = np.stack(all_targets)

        if self.use_calibration and self.calibrator is not None:
            probs_stack = self._calibrate_probs(probs_stack)

        probs_flat, targets_flat = self._filter_valid_pixels(probs_stack, targets_stack)

        if len(probs_flat) == 0:
            return {
                "probs": probs_stack,
                "targets": targets_stack,
                "metrics": {"csi": 0.0, "mcc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
                "threshold": 0.5,
                "n_samples": len(indices),
            }

        if fixed_threshold is not None:
            from evaluation.metrics import compute_metrics
            preds = (probs_flat >= fixed_threshold).astype(np.int32)
            tp = np.sum((preds == 1) & (targets_flat == 1))
            fp = np.sum((preds == 1) & (targets_flat == 0))
            fn = np.sum((preds == 0) & (targets_flat == 1))
            tn = np.sum((preds == 0) & (targets_flat == 0))

            best_metrics = compute_metrics(tp, fp, fn, tn)
            best_thr = fixed_threshold
        else:
            best_thr, best_metrics = self._optimize_threshold(probs_flat, targets_flat)

        return {
            "probs": probs_stack,
            "targets": targets_stack,
            "metrics": best_metrics,
            "threshold": best_thr,
            "n_samples": len(indices),
        }

    def run_inference(self) -> Dict:
        """
        Run inference on the test period.

        The decision threshold is always resolved BEFORE looking at the test
        predictions, using (in order of precedence):
          1. An explicit ``fixed_threshold`` (e.g. ``--threshold``).
          2. A recalibration performed exclusively on the validation split,
             if explicitly requested (``use_validation_fallback`` /
             ``--recalibrate``) — never on test.
          3. By default, the threshold already fixed on validation during
             grid search / model selection and stored in the checkpoint
             (``best_threshold``).
        The test split's own predictions are used only to compute the final
        metrics — never to choose the threshold.

        Returns:
            Dict with predictions, metrics, and configuration.
        """
        self.logger.header(f"INFERENCE - p={self.p}, q={self.q}")

        self.logger.info(f"Calibration: {'✅ ENABLED' if self.use_calibration else '❌ DISABLED'}")
        self.logger.info(f"Validation recalibration: {'✅ ENABLED' if self.use_validation_fallback else '❌ DISABLED'}")

        if self.spi_test is None:
            self.logger.error("SPI test data not available")
            return {"success": False}

        # =====================================================================
        # RESOLVE THE DECISION THRESHOLD (never using test data)
        # =====================================================================
        calibration_metrics = None
        used_recalibration = False

        if self.fixed_threshold is not None:
            threshold_to_apply = self.fixed_threshold
            self.logger.info(f"🔒 Fixed threshold: {threshold_to_apply:.3f} (validation recalibration SKIPPED)")
        elif self.use_validation_fallback and self.spi_val is not None:
            self.logger.info("Recalibrating threshold on the validation period only...")
            val_result = self.predict_subset(
                self.val_data,
                self.val_months,
                self.spi_val,
                fixed_threshold=None,  # Optimize exclusively on validation
            )
            if val_result["probs"] is not None:
                threshold_to_apply = val_result["threshold"]
                calibration_metrics = val_result["metrics"]
                used_recalibration = True
                self.logger.info(f"Validation-recalibrated threshold: {threshold_to_apply:.3f}")
                self.logger.info(f"Validation CSI: {calibration_metrics['csi']:.4f}")
                self.logger.info(f"Validation MCC: {calibration_metrics['mcc']:.4f}")
            else:
                threshold_to_apply = self.best_threshold
                self.logger.warning(
                    "Validation recalibration failed (no predictions); "
                    f"falling back to the checkpoint threshold: {threshold_to_apply:.3f}"
                )
        else:
            threshold_to_apply = self.best_threshold
            self.logger.info(
                f"Using the validation-selected threshold from the checkpoint: {threshold_to_apply:.3f}"
            )

        test_data_len = len(self.test_data)
        test_samples_possible = max(0, test_data_len - self.p - self.q + 1)

        self.logger.info(f"Test period: {len(self.test_indices)} months")
        self.logger.info(f"Possible samples: {test_samples_possible} (p={self.p}, q={self.q})")

        # Run prediction on test data with the threshold resolved above —
        # the test split is only ever used to compute metrics, never to
        # choose the threshold.
        test_result = self.predict_subset(
            self.test_data,
            self.test_months,
            self.spi_test,
            fixed_threshold=threshold_to_apply,
        )

        if test_result["probs"] is None:
            self.logger.error("No test samples available")
            return {"success": False}

        test_result["used_fallback"] = used_recalibration
        test_result["calibration_metrics"] = calibration_metrics

        self.logger.info(f"Test samples: {test_result['n_samples']}")

        self.logger.success("Inference complete")
        self.logger.info(f"  Samples: {test_result['n_samples']}")
        self.logger.info(f"  Threshold: {test_result['threshold']:.3f}")
        self.logger.info(f"  CSI: {test_result['metrics']['csi']:.4f}")
        self.logger.info(f"  MCC: {test_result['metrics']['mcc']:.4f}")

        if used_recalibration:
            self.logger.info("  Threshold source: validation recalibration")
        elif self.fixed_threshold is not None:
            self.logger.info(f"  Threshold source: fixed ({self.fixed_threshold:.3f})")
        else:
            self.logger.info("  Threshold source: checkpoint (validation-selected during grid search)")

        return test_result

    def save_rasters(self, result: Dict):
        """
        Save prediction rasters as GeoTIFF.

        Args:
            result: Result dict from run_inference().
        """
        if result is None or result.get("probs") is None:
            self.logger.error("No results to save")
            return

        try:
            import rasterio
        except ImportError:
            self.logger.warning("rasterio not available. Saving as numpy...")
            self._save_as_numpy(result)
            return

        probs = result["probs"]
        targets = result["targets"]
        threshold = result["threshold"]

        suffix = "_fallback" if result.get("used_fallback", False) else ""

        pred_dir = self.paths.get("inference_pred", self.paths["inference_dir"] / f"pred{suffix}")
        truth_dir = self.paths.get("inference_truth", self.paths["inference_dir"] / f"truth{suffix}")
        prob_dir = self.paths.get("inference_prob", self.paths["inference_dir"] / f"prob{suffix}")

        pred_dir.mkdir(parents=True, exist_ok=True)
        truth_dir.mkdir(parents=True, exist_ok=True)
        prob_dir.mkdir(parents=True, exist_ok=True)

        prob_profile, binary_profile = self._create_raster_profiles(probs.shape[1:])
        valid_mask = self._get_valid_mask(probs.shape[1:])

        self.logger.info(f"🔧 Post-processing: min_area={self.min_area}, hole_area={self.hole_area}")

        saved_count = 0
        for i in range(len(probs)):
            date_info = self._get_date_info(i)
            if date_info is None:
                continue
            year, month = date_info

            prob = probs[i].astype(np.float32)
            prob[~valid_mask] = prob_profile['nodata']
            self._save_raster(prob_dir / f"prob_{year}_{month:02d}.tif", prob, prob_profile)

            binary = (prob >= threshold).astype(np.uint8)
            if binary.sum() > 0:
                binary = postprocess_binary_mask(
                    binary.astype(np.float32),
                    threshold=0.5,
                    min_area=self.min_area,
                    hole_area=self.hole_area
                )
            binary[~valid_mask] = 255
            self._save_raster(pred_dir / f"pred_{year}_{month:02d}.tif", binary, binary_profile)

            truth = targets[i].astype(np.uint8)
            truth[~valid_mask] = 255
            self._save_raster(truth_dir / f"truth_{year}_{month:02d}.tif", truth, binary_profile)

            saved_count += 1

        if saved_count > 0:
            self.logger.success(f"Rasters saved to: {pred_dir.parent}")
            self.logger.info(f"  {saved_count} files generated")
            self.logger.info("  nodata=255 for binaries (0=non-drought, 1=drought, 255=invalid)")
            self.logger.info(f"  Post-processing: min_area={self.min_area}, hole_area={self.hole_area}")
        else:
            self.logger.warning("No rasters were saved")

    def _create_raster_profiles(self, shape: Tuple[int, int]) -> Tuple[Dict, Dict]:
        """Create raster profiles for the probability and binary rasters."""
        H, W = shape

        prob_profile = {
            'driver': 'GTiff',
            'height': H,
            'width': W,
            'count': 1,
            'dtype': 'float32',
            'nodata': -9999.0,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        }

        binary_profile = {
            'driver': 'GTiff',
            'height': H,
            'width': W,
            'count': 1,
            'dtype': 'uint8',
            'nodata': 255,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        }

        if 'crs' in self.metadata:
            prob_profile['crs'] = self.metadata['crs']
            binary_profile['crs'] = self.metadata['crs']
        if 'transform' in self.metadata:
            # Without this, rasterio defaults to the identity transform, so
            # every saved raster carries a valid CRS but no real geographic
            # extent - axes on any figure built from these files (lon/lat
            # via _get_geo_coords) end up showing raw pixel indices mislabeled
            # as degrees, even though the CRS metadata looks correct.
            prob_profile['transform'] = self.metadata['transform']
            binary_profile['transform'] = self.metadata['transform']

        return prob_profile, binary_profile

    def _get_valid_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Get the validity mask resized to the given shape."""
        if self.valid_mask is not None:
            if self.valid_mask.shape != shape:
                from skimage.transform import resize
                return resize(
                    self.valid_mask.astype(np.float32),
                    shape,
                    order=0,
                    preserve_range=True
                ).astype(bool)
            return self.valid_mask
        return np.ones(shape, dtype=bool)

    def _get_date_info(self, idx: int) -> Optional[Tuple[int, int]]:
        """
        Get (year, month) for a given sample index in the test period.

        `predict_subset()` builds sample `idx` from the local index
        `t = self.p + idx` within `test_data` (since its `indices` range
        over `range(self.p, T - self.q + 1)`), with the target at
        `t + self.q - 1` (window ends at `t-1`; target is the qth month
        after that window — same convention as ClimateDataset). The global
        target index therefore has to add `p`, `idx`, and `q - 1`, or every
        saved raster ends up labeled one month later than it should.
        """
        target_local_idx = self.p + idx + self.q - 1
        if target_local_idx < len(self.test_indices):
            global_idx = self.test_indices[target_local_idx]
            if global_idx < len(self.months):
                return self.years[global_idx], self.months[global_idx]
        return None

    def _save_raster(self, path: Path, data: np.ndarray, profile: Dict):
        """Save a raster as GeoTIFF."""
        import rasterio
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(data, 1)

    def _save_as_numpy(self, result: Dict):
        """Fallback: save results as numpy files when rasterio is unavailable."""
        probs = result["probs"]
        targets = result["targets"]
        threshold = result["threshold"]

        suffix = "_fallback" if result.get("used_fallback", False) else ""
        output_dir = self.paths["inference_dir"] / f"numpy_output{suffix}"
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "probs.npy", probs)
        np.save(output_dir / "targets.npy", targets)
        np.save(output_dir / "binary.npy", (probs >= threshold).astype(np.uint8))

        if self.valid_mask is not None:
            np.save(output_dir / "valid_mask.npy", self.valid_mask)

        meta = {
            "n_samples": len(probs),
            "threshold": float(threshold),
            "p": self.p,
            "q": self.q,
            "region": self.config.region,
            "used_fallback": result.get("used_fallback", False),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        self.logger.success(f"Results saved as numpy: {output_dir}")