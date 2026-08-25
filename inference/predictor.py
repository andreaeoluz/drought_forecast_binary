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
    """Handles inference on test period with optional calibration and fallback."""

    def __init__(
        self,
        config: ExperimentConfig,
        base_data_path: Path,
        p: int,
        q: int,
        model_type: str = "pretrained",
        optimization_metric: str = None,  # ← None = usa config
        use_calibration: bool = None,     # ← None = usa config
        use_validation_fallback: bool = True,
        fixed_threshold: Optional[float] = None,
    ):
        """
        Initialize inference predictor.

        Args:
            config: Experiment configuration
            base_data_path: Path to data directory
            p: History length
            q: Forecast horizon
            model_type: 'pretrained' or 'scratch'
            optimization_metric: Metric for threshold optimization ('mcc' or 'csi').
                If None, uses config.optimization.primary_metric.
            use_calibration: Whether to apply probability calibration.
                If None, uses config.training.calibrate.
            use_validation_fallback: Whether to use validation for threshold calibration
            fixed_threshold: If provided, use this threshold instead of optimizing
        """
        self.config = config
        self.base_data_path = base_data_path
        self.p = p
        self.q = q
        self.model_type = model_type
        
        # ✅ Métrica de otimização (com fallback para config)
        if optimization_metric is None:
            self.optimization_metric = config.optimization.primary_metric
        else:
            self.optimization_metric = optimization_metric
        
        # ✅ Calibração (com fallback para config)
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
        
        # ✅ Log das configurações
        self.logger.info(f"📊 InferencePredictor initialized:")
        self.logger.info(f"   Optimization metric: {self.optimization_metric.upper()}")
        self.logger.info(f"   Calibration: {'✅ ENABLED' if self.use_calibration else '❌ DISABLED'}")
        self.logger.info(f"   Validation fallback: {'✅ ENABLED' if self.use_validation_fallback else '❌ DISABLED'}")
        if self.fixed_threshold is not None:
            self.logger.info(f"   Fixed threshold: {self.fixed_threshold:.3f}")

    def _setup_postprocessing(self):
        """Setup spatial post-processing parameters."""
        factor = self.config.data.downsample_h
        min_area_original = getattr(self.config, 'min_area_original_pixels', 10)
        min_area = max(1, min_area_original // (factor ** 2))
        self.min_area = min_area
        self.hole_area = max(1, self.min_area // 2)

    def _load_data(self):
        """Load climate data, SPI and normalizer."""
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
        """Resize validity mask to match data dimensions."""
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
        """Load and resize SPI data."""
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
        """Load climate normalizer."""
        normalizer_path = self.paths["autoencoder_dir"] / "normalizer.json"
        if not normalizer_path.exists():
            normalizer_path = self.paths["grid_search_dir"] / "normalizer.json"
        if not normalizer_path.exists():
            raise FileNotFoundError(f"Normalizer not found: {normalizer_path}")
        self.normalizer = ClimateNormalizer.load(normalizer_path)
        self.logger.success(f"Normalizer loaded: {normalizer_path}")

    def _setup_temporal_indices(self):
        """Setup temporal indices for test and validation periods."""
        self.time_idx = np.array([y * 12 + (m - 1) for y, m in zip(self.years, self.months)])
        split = self.config.split

        # Test period
        test_start = split.ym_to_int(split.test[0])
        test_end = split.ym_to_int(split.test[1])
        self.test_mask = (self.time_idx >= test_start) & (self.time_idx <= test_end)
        self.test_indices = np.where(self.test_mask)[0]
        self.test_data = self.data[self.test_mask]
        self.test_months = self.months[self.test_mask]
        self.spi_test = self.spi[self.test_mask] if self.spi is not None else None

        # Validation period (GS validation)
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
        """Find the model checkpoint path."""
        model_filename = f"model_p{self.p}_q{self.q}.pth"
        candidates = [
            self.paths["grid_search_pretrained"] / model_filename,
            self.paths["grid_search_scratch"] / model_filename,
            self.paths["grid_search_dir"] / self.model_type / model_filename,
            self.paths["grid_search_dir"] / model_filename,
        ]
        for path in candidates:
            if path.exists():
                self.logger.info(f"Model found: {path}")
                return path
        raise FileNotFoundError(
            f"Model not found for p={self.p}, q={self.q}, type={self.model_type}"
        )

    def _load_model(self):
        """Load the trained model."""
        checkpoint_path = self._find_model_path()
        self.logger.info(f"Loading model: {checkpoint_path}")
        model_config = self.config.get_model_config("predictor")
        self.model = ConvLSTMPredictor(model_config).to(self.config.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.eval()
        self.best_threshold = checkpoint.get("best_threshold", 0.20)
        self.best_csi = checkpoint.get("best_csi", 0.0)
        self.best_mcc = checkpoint.get("best_mcc", 0.0)
        self.logger.success(f"Model loaded (CSI: {self.best_csi:.4f}, MCC: {self.best_mcc:.4f})")

    def _load_calibrator(self):
        """Load probability calibrator if enabled."""
        if not self.use_calibration:
            return
        self.logger.info("Loading calibrator...")
        checkpoint_path = self._find_model_path()

        # Try to load from checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
            if "calibrator" in checkpoint and checkpoint["calibrator"] is not None:
                self.calibrator = checkpoint["calibrator"]
                self.logger.success("Calibrator loaded from checkpoint")
                return
        except Exception as e:
            self.logger.debug(f"Could not load calibrator from checkpoint: {e}")

        # Try to load from pickle file
        calibrator_paths = [
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
        """Apply calibration to probabilities."""
        if self.calibrator is None or not self.use_calibration:
            return probs

        # ✅ USAR O CALIBRADOR (seja Platt ou Isotonic)
        original_shape = probs.shape
        probs_flat = probs.flatten()
        
        # O calibrador já tem o método transform
        calibrated = self.calibrator.transform(probs_flat)
        return calibrated.reshape(original_shape)

    def _prepare_data(self, data_subset: np.ndarray, months_subset: np.ndarray) -> np.ndarray:
        """Prepare data for inference (normalize + transpose)."""
        data_norm = self.normalizer.transform(data_subset, months_subset, self.valid_mask)
        data_norm = np.nan_to_num(data_norm, nan=0.0)
        return np.transpose(data_norm, (0, 3, 1, 2))

    def _get_sample_indices(self, data_len: int) -> List[int]:
        """Get valid sample indices for a data subset."""
        min_required = self.p + self.q
        if data_len <= min_required:
            return []
        return list(range(self.p, data_len - self.q))

    def _filter_valid_pixels(self, probs_stack: np.ndarray, targets_stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter valid pixels using the validity mask."""
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
        """Optimize threshold using adaptive range."""
        thresholds = self.config.calibration_thresholds
        
        # Filtrar thresholds para o range relevante
        p1 = np.percentile(probs, 1)
        p99 = np.percentile(probs, 99)
        
        # Usar apenas thresholds dentro do range
        thresholds = [t for t in thresholds if p1 <= t <= p99]
        
        if len(thresholds) < 10:
            # Fallback para range adaptativo
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
            data_subset: Climate data (T, H, W, C)
            months_subset: Month indices (T,)
            spi_subset: SPI values (T, H, W)
            fixed_threshold: If provided, use this threshold instead of optimizing

        Returns:
            Dictionary with predictions, targets, metrics, and threshold
        """
        data_ch = self._prepare_data(data_subset, months_subset)
        T, C, H, W = data_ch.shape
        indices = self._get_sample_indices(T)

        if not indices:
            return {"probs": None, "targets": None, "metrics": None, "n_samples": 0}

        all_probs = []
        all_targets = []

        binary_mask = (spi_subset <= self.config.spi.threshold).astype(np.float32)

        with torch.no_grad():
            for t in indices:
                target_idx = t + self.q
                x_seq = data_ch[t - self.p:t]
                x_tensor = torch.from_numpy(x_seq).float().unsqueeze(0).to(self.config.device)
                logits, _ = self.model(x_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
                all_probs.append(probs)
                all_targets.append(binary_mask[target_idx])

        probs_stack = np.stack(all_probs)
        targets_stack = np.stack(all_targets)

        # Apply calibration if enabled
        if self.use_calibration and self.calibrator is not None:
            probs_stack = self._calibrate_probs(probs_stack)

        # Filter valid pixels
        probs_flat, targets_flat = self._filter_valid_pixels(probs_stack, targets_stack)

        if len(probs_flat) == 0:
            return {
                "probs": probs_stack,
                "targets": targets_stack,
                "metrics": {"csi": 0.0, "mcc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
                "threshold": 0.5,
                "n_samples": len(indices),
            }

        # Use fixed threshold if provided, otherwise optimize
        if fixed_threshold is not None:
            # Compute metrics at fixed threshold
            from evaluation.metrics import compute_metrics
            preds = (probs_flat > fixed_threshold).astype(np.int32)
            tp = np.sum((preds == 1) & (targets_flat == 1))
            fp = np.sum((preds == 1) & (targets_flat == 0))
            fn = np.sum((preds == 0) & (targets_flat == 1))
            tn = np.sum((preds == 0) & (targets_flat == 0))

            from evaluation.metrics import compute_metrics as compute_metrics_dict
            best_metrics = compute_metrics_dict(tp, fp, fn, tn)
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

        Returns:
            Dictionary with predictions, metrics, and configuration
        """
        self.logger.header(f"INFERENCE - p={self.p}, q={self.q}")

        # Log configuration
        self.logger.info(f"Calibration: {'✅ ENABLED' if self.use_calibration else '❌ DISABLED'}")
        self.logger.info(f"Validation fallback: {'✅ ENABLED' if self.use_validation_fallback else '❌ DISABLED'}")

        if self.fixed_threshold is not None:
            self.logger.info(f"🔒 Fixed threshold: {self.fixed_threshold:.3f} (optimization SKIPPED)")

        if self.spi_test is None:
            self.logger.error("SPI test data not available")
            return {"success": False}

        # Get test samples
        test_data_len = len(self.test_data)
        test_samples_possible = max(0, test_data_len - self.p - self.q + 1)

        self.logger.info(f"Test period: {len(self.test_indices)} months")
        self.logger.info(f"Possible samples: {test_samples_possible} (p={self.p}, q={self.q})")

        # Run prediction on test data
        test_result = self.predict_subset(
            self.test_data,
            self.test_months,
            self.spi_test,
            fixed_threshold=self.fixed_threshold
        )

        if test_result["probs"] is None:
            self.logger.error("No test samples available")
            return {"success": False}

        test_samples = test_result["n_samples"]
        self.logger.info(f"Test samples: {test_samples}")

        # Check if we should use validation fallback for threshold calibration
        min_samples_for_eval = 5
        use_fallback = (
            self.use_validation_fallback
            and test_samples < min_samples_for_eval
            and self.spi_val is not None
            and self.fixed_threshold is None  # ✅ Don't use fallback if threshold is fixed
        )

        if use_fallback:
            self.logger.warning(f"Test has only {test_samples} samples (< {min_samples_for_eval})")
            self.logger.info("Using validation period for threshold calibration...")

            val_result = self.predict_subset(
                self.val_data,
                self.val_months,
                self.spi_val,
                fixed_threshold=None  # Optimize on validation
            )

            if val_result["probs"] is None:
                self.logger.warning("Validation predictions failed. Using test threshold.")
                final_threshold = test_result["threshold"]
                calibration_metrics = None
            else:
                final_threshold = val_result["threshold"]
                calibration_metrics = val_result["metrics"]
                self.logger.info(f"Calibrated threshold: {final_threshold:.3f}")
                self.logger.info(f"Calibration CSI: {calibration_metrics['csi']:.4f}")
                self.logger.info(f"Calibration MCC: {calibration_metrics['mcc']:.4f}")

            test_result["threshold"] = final_threshold
            test_result["calibration_metrics"] = calibration_metrics
            test_result["used_fallback"] = True
            test_result["validation_samples"] = val_result["n_samples"] if val_result["probs"] is not None else 0

            # ✅ Recompute test metrics with final threshold
            if self.fixed_threshold is None:
                probs_flat, targets_flat = self._filter_valid_pixels(
                    test_result["probs"], test_result["targets"]
                )
                if len(probs_flat) > 0:
                    from evaluation.metrics import compute_metrics
                    preds = (probs_flat > final_threshold).astype(np.int32)
                    tp = np.sum((preds == 1) & (targets_flat == 1))
                    fp = np.sum((preds == 1) & (targets_flat == 0))
                    fn = np.sum((preds == 0) & (targets_flat == 1))
                    tn = np.sum((preds == 0) & (targets_flat == 0))
                    test_result["metrics"] = compute_metrics(tp, fp, fn, tn)

        else:
            test_result["used_fallback"] = False
            test_result["calibration_metrics"] = None

        # Log results
        self.logger.success("Inference complete")
        self.logger.info(f"  Samples: {test_result['n_samples']}")
        self.logger.info(f"  Threshold: {test_result['threshold']:.3f}")
        self.logger.info(f"  CSI: {test_result['metrics']['csi']:.4f}")
        self.logger.info(f"  MCC: {test_result['metrics']['mcc']:.4f}")

        if test_result.get("used_fallback", False):
            self.logger.info(f"  Fallback used: YES (validation samples: {test_result.get('validation_samples', 0)})")

        if self.fixed_threshold is not None:
            self.logger.info(f"  Fixed threshold: {self.fixed_threshold:.3f} (applied)")

        return test_result

    def save_rasters(self, result: Dict):
        """
        Save prediction rasters as GeoTIFF.

        Args:
            result: Result dictionary from run_inference()
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

            # Probability raster
            prob = probs[i].astype(np.float32)
            prob[~valid_mask] = prob_profile['nodata']
            self._save_raster(prob_dir / f"prob_{year}_{month:02d}.tif", prob, prob_profile)

            # Binary prediction with post-processing
            binary = (prob > threshold).astype(np.uint8)
            if binary.sum() > 0:
                binary = postprocess_binary_mask(
                    binary.astype(np.float32),
                    threshold=0.5,
                    min_area=self.min_area,
                    hole_area=self.hole_area
                )
            binary[~valid_mask] = 255
            self._save_raster(pred_dir / f"pred_{year}_{month:02d}.tif", binary, binary_profile)

            # Ground truth
            truth = targets[i].astype(np.uint8)
            truth[~valid_mask] = 255
            self._save_raster(truth_dir / f"truth_{year}_{month:02d}.tif", truth, binary_profile)

            saved_count += 1

        if saved_count > 0:
            self.logger.success(f"Rasters saved to: {pred_dir.parent}")
            self.logger.info(f"  {saved_count} files generated")
            self.logger.info(f"  nodata=255 for binaries (0=drought, 1=non-drought, 255=invalid)")
            self.logger.info(f"  Post-processing: min_area={self.min_area}, hole_area={self.hole_area}")
        else:
            self.logger.warning("No rasters were saved")

    def _create_raster_profiles(self, shape: Tuple[int, int]) -> Tuple[Dict, Dict]:
        """Create raster profiles for probability and binary rasters."""
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

        return prob_profile, binary_profile

    def _get_valid_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Get validity mask resized to the given shape."""
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
        """Get (year, month) for a given index in the test period."""
        test_indices = self.test_indices
        if idx < len(test_indices):
            global_idx = test_indices[idx] + self.q
            if global_idx < len(self.months):
                return self.years[global_idx], self.months[global_idx]
        return None

    def _save_raster(self, path: Path, data: np.ndarray, profile: Dict):
        """Save a raster as GeoTIFF."""
        import rasterio
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(data, 1)

    def _save_as_numpy(self, result: Dict):
        """Fallback: save results as numpy files."""
        probs = result["probs"]
        targets = result["targets"]
        threshold = result["threshold"]

        suffix = "_fallback" if result.get("used_fallback", False) else ""
        output_dir = self.paths["inference_dir"] / f"numpy_output{suffix}"
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "probs.npy", probs)
        np.save(output_dir / "targets.npy", targets)
        np.save(output_dir / "binary.npy", (probs > threshold).astype(np.uint8))

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