"""predictor.py - Predictor trainer."""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict

from .base import BaseTrainer
from .losses import build_loss
from utils.logger import Logger, Colors
from evaluation.metrics import find_best_threshold


class PredictorTrainer(BaseTrainer):
    """Trainer for ConvLSTM Predictor."""

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config,
        save_path: Path,
        freeze_encoder_first_epoch: bool = False,
        optimization_metric: str = None,
        calibrate: bool = None,
    ):
        super().__init__(model, train_dataset, val_dataset, config, save_path)

        self.logger = Logger()

        self.freeze_encoder_first_epoch = freeze_encoder_first_epoch
        self.freeze_encoder_epochs = getattr(config.training, 'freeze_encoder_epochs', 5)
        self.encoder_lr_factor = getattr(config.training, 'encoder_lr_factor', 0.1)

        self.optimization_metric = (
            optimization_metric or config.optimization.primary_metric
        )

        self.calibrate = (
            calibrate if calibrate is not None
            else getattr(config.training, 'calibrate', False)
        )
        self.calibrator = None

        pos_weight = self._compute_pos_weight()

        alpha = config.loss.alpha
        if config.loss.use_dynamic_alpha and self._prevalence is not None:
            alpha = 1.0 - self._prevalence * 2
            alpha = max(0.25, min(0.75, alpha))

        self.criterion = build_loss(
            loss_name=config.loss.name,
            pos_weight=pos_weight,
            gamma=config.loss.gamma,
            alpha=alpha,
            prevalence=self._prevalence,
            use_dynamic_alpha=config.loss.use_dynamic_alpha,
        ).to(self.device)

        self.loss_mask = None
        if hasattr(train_dataset, 'valid_mask') and train_dataset.valid_mask is not None:
            self.loss_mask = torch.from_numpy(train_dataset.valid_mask.astype(np.float32))
            self.loss_mask = self.loss_mask.to(self.device).unsqueeze(0).unsqueeze(0)

        # ✅ CRIAR OPTIMIZER UMA VEZ (não recriar depois)
        self.optimizer = self._create_optimizer()

        self.logger.info(f"📊 PredictorTrainer initialized:")
        self.logger.info(f"   Optimization metric: {self.optimization_metric.upper()}")
        self.logger.info(f"   Calibrate: {self.calibrate}")
        self.logger.info(f"   Freeze epochs: {self.freeze_encoder_epochs if self.freeze_encoder_first_epoch else 0}")
        self.logger.info(f"   Encoder LR factor: {self.encoder_lr_factor}")

    def _compute_pos_weight(self) -> float:
        total = 0
        positives = 0

        max_samples = min(1000, len(self.train_dataset))
        for i in range(max_samples):
            batch = self.train_dataset[i]
            y = batch["y_bin"]
            if "mask" in batch:
                mask = batch["mask"]
                y_masked = y * mask
                total += mask.sum().item()
                positives += y_masked.sum().item()
            else:
                total += y.numel()
                positives += y.sum().item()

        if positives == 0:
            self._prevalence = 0.0
            return 1.0
        if total == positives:
            self._prevalence = 1.0
            return 1.0

        prevalence = positives / total
        self._prevalence = prevalence

        negatives = total - positives
        raw_pos_weight = negatives / positives

        pos_weight = raw_pos_weight ** 0.25
        pos_weight = np.clip(pos_weight, 1.0, 3.0)

        return pos_weight

    def _create_optimizer(self):
        """
        Cria otimizador com LRs diferenciados para encoder/attention e decoder.
        
        ✅ Mantém LRs diferenciados durante TODO o treinamento.
        ✅ Atenção usa LR do decoder (não é transferida)
        """
        # Se não for TL, usar LR único para todos
        if not self.freeze_encoder_first_epoch:
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )

        # Separar parâmetros por grupo
        encoder_params = []
        attention_params = []
        decoder_params = []

        for name, param in self.model.named_parameters():
            if name.startswith('encoder'):
                encoder_params.append(param)
            elif name.startswith('attention'):
                attention_params.append(param)
            else:
                decoder_params.append(param)

        encoder_lr = self.config.training.learning_rate * self.encoder_lr_factor
        decoder_lr = self.config.training.learning_rate

        param_groups = []

        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': encoder_lr,
                'name': 'encoder'
            })

        # ✅ Atenção: NÃO é transferida, usa LR do decoder
        if attention_params:
            param_groups.append({
                'params': attention_params,
                'lr': decoder_lr,  # ← Atenção usa LR do decoder!
                'name': 'attention (random)'
            })

        if decoder_params:
            param_groups.append({
                'params': decoder_params,
                'lr': decoder_lr,
                'name': 'decoder'
            })

        self.logger.info(f"   Optimizer groups:")
        for group in param_groups:
            self.logger.info(f"      {group['name']}: LR={group['lr']:.2e}")

        return torch.optim.Adam(
            param_groups,
            weight_decay=self.config.training.weight_decay,
        )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            x = batch["x"].to(self.device)
            y_bin = batch["y_bin"].to(self.device).float().unsqueeze(1)

            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad(set_to_none=True)
            logits, _ = self.model(x)

            loss_raw = self.criterion(logits, y_bin)

            if mask is not None:
                masked_loss = loss_raw * mask
                valid_pixels = mask.sum()
                if valid_pixels > 0:
                    loss = masked_loss.sum() / valid_pixels
                else:
                    loss = loss_raw.mean()
            elif self.loss_mask is not None and loss_raw.dim() == 4:
                loss = (loss_raw * self.loss_mask).sum() / (self.loss_mask.sum() + 1e-8)
            else:
                loss = loss_raw.mean()

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                total_loss += loss.item() * x.size(0)
                n_samples += x.size(0)

        return total_loss / n_samples if n_samples > 0 else float('inf')

    def _collect_predictions(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_probs = []
        all_targets = []
        all_masks = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch["x"].to(self.device)
                y_bin = batch["y_bin"].to(self.device).float().unsqueeze(1)

                mask = batch.get("mask", None)
                if mask is not None:
                    all_masks.append(mask.cpu().numpy().flatten())

                logits, _ = self.model(x)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                all_probs.extend(probs)
                all_targets.extend(y_bin.cpu().numpy().flatten())

        probs = np.array(all_probs)
        targets = np.array(all_targets)

        if all_masks:
            masks = np.concatenate(all_masks)
            valid_idx = masks > 0.5
            probs = probs[valid_idx]
            targets = targets[valid_idx]

        valid = ~(np.isnan(probs) | np.isnan(targets))
        probs = probs[valid]
        targets = targets[valid]

        return probs, targets

    def fit_calibrator(self, train_loader, val_loader) -> None:
        from evaluation.calibration import PlattCalibrator, IsotonicCalibrator

        calibrator_type = getattr(self.config.training, 'calibrator_type', 'platt')

        if calibrator_type == "isotonic":
            self.logger.info("  📊 Fitting calibrator (Isotonic Regression)...")
            self.calibrator = IsotonicCalibrator()
        else:
            self.logger.info("  📊 Fitting calibrator (Platt Scaling)...")
            self.calibrator = PlattCalibrator(C=0.1, random_state=42)

        train_probs, train_targets = self._collect_predictions(train_loader)
        val_probs, val_targets = self._collect_predictions(val_loader)

        if len(train_probs) == 0 or len(val_probs) == 0:
            self.logger.warning("  ⚠️ No predictions collected for calibration!")
            return

        probs = np.concatenate([train_probs, val_probs])
        targets = np.concatenate([train_targets, val_targets])

        self.logger.info(f"  📊 Calibration data: {len(probs):,} samples")
        self.logger.info(f"     Positive rate: {targets.mean():.4f}")
        self.logger.info(f"     Prob range: {probs.min():.4f} - {probs.max():.4f}")

        self.calibrator.fit(probs, targets)

        probs_calibrated = self.calibrator.transform(probs)
        self.logger.info(f"  📊 Calibration results:")
        self.logger.info(f"     Mean prob (raw): {probs.mean():.4f}")
        self.logger.info(f"     Mean prob (calibrated): {probs_calibrated.mean():.4f}")
        self.logger.info(f"     Positive rate: {targets.mean():.4f}")

        try:
            import joblib
            calib_path = self.save_path.parent / "calibrator.pkl"
            joblib.dump(self.calibrator, calib_path)
            self.logger.success(f"  ✅ Calibrator saved to: {calib_path}")
        except Exception as e:
            self.logger.warning(f"  ⚠️ Could not save calibrator: {e}")

    def calibrate_probs(self, probs: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return probs

        original_shape = probs.shape
        probs_flat = probs.flatten()

        if hasattr(self.calibrator, 'transform'):
            calibrated = self.calibrator.transform(probs_flat)
        else:
            from sklearn.linear_model import LogisticRegression
            if isinstance(self.calibrator, LogisticRegression):
                calibrated = self.calibrator.predict_proba(probs_flat.reshape(-1, 1))[:, 1]
            else:
                calibrated = probs_flat

        return calibrated.reshape(original_shape)

    def validate(self, val_loader, use_calibration: bool = True):
        probs, targets = self._collect_predictions(val_loader)

        if len(probs) == 0:
            return {"threshold": 0.5, "metrics": {"csi": 0.0, "mcc": 0.0}, "csi": 0.0, "mcc": 0.0}

        if use_calibration and self.calibrator is not None:
            probs_original = probs.copy()
            probs = self.calibrate_probs(probs)
            self.logger.debug(f"  📊 Calibrated: {probs_original.mean():.4f} → {probs.mean():.4f}")

        if hasattr(self.config, 'calibration_thresholds') and self.config.calibration_thresholds:
            thresholds = np.array(self.config.calibration_thresholds)
        else:
            thresholds = np.arange(0.01, 0.99, 0.005)

        p1 = np.percentile(probs, 1)
        p99 = np.percentile(probs, 99)
        thresholds = thresholds[(thresholds >= p1) & (thresholds <= p99)]

        if len(thresholds) < 10:
            low = max(0.001, p1 - 0.02)
            high = min(0.999, p99 + 0.02)
            thresholds = np.arange(low, high + 0.005, 0.005)

        best_thr, best_metrics = find_best_threshold(
            probs, targets, thresholds,
            metric=self.optimization_metric,
        )

        best_thr = max(best_thr, 0.05)

        return {
            "threshold": best_thr,
            "metrics": best_metrics,
            "csi": best_metrics["csi"],
            "mcc": best_metrics["mcc"],
            "score": best_metrics[self.optimization_metric],
        }

    def train(self):
        self.logger.header("TREINAMENTO DO PREDICTOR")
        self.logger.info(f"Métrica de otimização: {self.optimization_metric.upper()}")

        # ✅ Log da estratégia de TL
        if self.freeze_encoder_first_epoch:
            self.logger.info(f"🔹 Estratégia: TL somente Encoder (Attention aleatória)")
        else:
            self.logger.info(f"🔹 Estratégia: Scratch (tudo aleatório)")

        train_loader, val_loader = self.create_dataloaders(shuffle_train=True)

        best_score = -1.0
        patience_counter = 0
        min_epochs = 10

        if self.freeze_encoder_first_epoch:
            self.model.freeze_encoder(freeze=True)
            self.logger.info(f"🔒 Encoder congelado por {self.freeze_encoder_epochs} épocas")
            self.logger.info(f"   Attention NÃO está congelada (aleatória)")

        for epoch in range(self.config.training.epochs):
            # ✅ Descongelar apenas o encoder (attention já estava livre)
            if self.freeze_encoder_first_epoch and epoch == self.freeze_encoder_epochs:
                self.model.freeze_encoder(freeze=False)
                self.logger.info(f"🔓 Encoder descongelado na época {epoch}")
                self.logger.info(f"   Mantendo LRs: Encoder={self.config.training.learning_rate * self.encoder_lr_factor:.2e}, Decoder={self.config.training.learning_rate:.2e}")

            train_loss = self.train_epoch(train_loader)

            val_results = self.validate(val_loader, use_calibration=False)

            epoch_score = val_results["score"]
            epoch_csi = val_results["csi"]
            epoch_mcc = val_results["mcc"]

            improved = epoch_score > best_score
            if improved:
                best_score = epoch_score
                patience_counter = 0
                self.save_checkpoint(epoch, {
                    "best_threshold": val_results["threshold"],
                    "best_score": best_score,
                    "best_csi": epoch_csi,
                    "best_mcc": epoch_mcc,
                    "precision": val_results["metrics"]["precision"],
                    "recall": val_results["metrics"]["recall"],
                    "f1": val_results["metrics"]["f1"],
                    "far": val_results["metrics"].get("far", 0.0),
                })
            else:
                patience_counter += 1

            metric_label = self.optimization_metric.upper()
            metric_value = epoch_mcc if self.optimization_metric == "mcc" else epoch_csi

            status = f"{Colors.GREEN}✓{Colors.RESET}" if improved else f"ES {patience_counter}/{self.config.training.patience}"

            self.logger.info(
                f"Época {epoch:3d} | Loss {train_loss:.4f} | "
                f"{metric_label} {metric_value:.4f} | "
                f"CSI {epoch_csi:.4f} | MCC {epoch_mcc:.4f} | "
                f"Thr {val_results['threshold']:.3f} | "
                f"FAR {val_results['metrics'].get('far', 0.0):.3f} | {status}"
            )

            if self.freeze_encoder_first_epoch and epoch < self.freeze_encoder_epochs:
                self.logger.debug(f"   🔒 Encoder congelado (época {epoch+1}/{self.freeze_encoder_epochs})")

            if epoch >= min_epochs and patience_counter >= self.config.training.patience:
                self.logger.warning(f"Early stopping na época {epoch}")
                break

        checkpoint = torch.load(self.save_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.calibrate:
            self.logger.info("📊 Training calibrator on validation data...")
            self.fit_calibrator(train_loader, val_loader)

        if self.calibrate and self.calibrator is not None:
            self.logger.info("📊 Evaluating with calibration...")
            final_val_calibrated = self.validate(val_loader, use_calibration=True)

            checkpoint["best_threshold"] = final_val_calibrated["threshold"]
            checkpoint["best_score"] = final_val_calibrated["score"]
            checkpoint["best_csi"] = final_val_calibrated["csi"]
            checkpoint["best_mcc"] = final_val_calibrated["mcc"]
            checkpoint["final_metrics"] = final_val_calibrated["metrics"]
            checkpoint["calibrator"] = self.calibrator

            self.logger.success(f"✅ Calibrated - CSI: {final_val_calibrated['csi']:.4f} | MCC: {final_val_calibrated['mcc']:.4f}")

        else:
            final_val = self.validate(val_loader, use_calibration=False)
            checkpoint["best_threshold"] = final_val["threshold"]
            checkpoint["best_score"] = final_val["score"]
            checkpoint["best_csi"] = final_val["csi"]
            checkpoint["best_mcc"] = final_val["mcc"]
            checkpoint["final_metrics"] = final_val["metrics"]
            checkpoint["calibrator"] = None

            self.logger.success(f"✅ Uncalibrated - CSI: {final_val['csi']:.4f} | MCC: {final_val['mcc']:.4f}")

        torch.save(checkpoint, self.save_path)
        self.logger.success(f"✅ Final checkpoint saved")

        return best_score