"""trainer.py - Classes base para treinamento"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import random
from sklearn.isotonic import IsotonicRegression

from utils import set_reproducible_seeds
from utils.logger import Logger, Colors
from evaluation.metrics import find_best_threshold
from training.losses import build_loss


class BaseTrainer:
    """Classe base para treinamento de modelos."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        val_dataset,
        config,
        save_path: Path,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.save_path = save_path
        self.device = config.device
        self.model.to(self.device)
        
        set_reproducible_seeds(config.random_seed, deterministic=True)
    
    def create_dataloaders(self, shuffle_train: bool = True):
        """Cria DataLoaders determinísticos."""
        batch_size = self.config.training.batch_size
        seed = self.config.random_seed
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        def worker_init_fn(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        
        if shuffle_train and hasattr(self.train_dataset, 'sample_weights') and self.train_dataset.sample_weights is not None:
            weights = torch.as_tensor(self.train_dataset.sample_weights, dtype=torch.double)
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True, generator=g)
            train_loader = torch.utils.data.DataLoader(
                self.train_dataset, 
                batch_size=batch_size, 
                sampler=sampler, 
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=worker_init_fn,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                self.train_dataset, 
                batch_size=batch_size, 
                shuffle=shuffle_train, 
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=worker_init_fn,
                generator=g,
            )
        
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )
        
        return train_loader, val_loader
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, Any]):
        """Salva checkpoint do modelo."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            **metrics,
        }
        torch.save(checkpoint, self.save_path)
    
    def load_checkpoint(self, checkpoint_path: Path, strict: bool = True):
        """Carrega checkpoint do modelo."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        return checkpoint


class PredictorTrainer(BaseTrainer):
    """Trainer para ConvLSTM Predictor."""
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config,
        save_path: Path,
        freeze_encoder_first_epoch: bool = False,
        optimization_metric: str = "csi",
        calibrate: bool = True,
    ):
        super().__init__(model, train_dataset, val_dataset, config, save_path)
        
        self.freeze_encoder_first_epoch = freeze_encoder_first_epoch
        self.freeze_encoder_epochs = getattr(config.training, 'freeze_encoder_epochs', 5)
        self.encoder_lr_factor = getattr(config.training, 'encoder_lr_factor', 0.1)
        self.optimization_metric = optimization_metric
        self.calibrate = calibrate
        self.calibrator = None  # Será treinado após o treinamento
        
        self.logger = Logger()
        
        pos_weight = self._compute_pos_weight()
        prevalence = self._prevalence
        
        if hasattr(self.model, 'prevalence'):
            self.model.prevalence = prevalence
        
        # Alpha da Focal Loss - agora com limites mais conservadores
        if config.loss.use_dynamic_alpha and prevalence is not None:
            # Para classes raras, alpha deve ser mais baixo para evitar overconfidence
            alpha = 1.0 - prevalence * 2
            alpha = max(0.25, min(0.75, alpha))
            
            self.logger.debug(
                f"Alpha balanceado: {alpha:.3f} (prevalência: {prevalence:.2%})"
            )
        else:
            alpha = config.loss.alpha if config.loss.alpha is not None else 0.5
        
        self.criterion = build_loss(
            loss_name="focal",
            pos_weight=pos_weight,
            gamma=config.loss.gamma,
            alpha=alpha
        ).to(self.device)
        
        # ============================================================
        # ✅ CORREÇÃO: Obter máscara de validade do dataset
        # A máscara é usada apenas para a perda, não para os dados
        # ============================================================
        self.loss_mask = None
        if hasattr(train_dataset, 'valid_mask') and train_dataset.valid_mask is not None:
            self.loss_mask = torch.from_numpy(train_dataset.valid_mask.astype(np.float32))
            self.loss_mask = self.loss_mask.to(self.device)
            # Expandir para (1, 1, H, W) para broadcasting
            self.loss_mask = self.loss_mask.unsqueeze(0).unsqueeze(0)
            
            valid_pixels = train_dataset.valid_mask.sum()
            total_pixels = train_dataset.valid_mask.size
            self.logger.info(f"📊 Máscara de perda: {valid_pixels:,} pixels válidos / {total_pixels:,} total")
        else:
            self.logger.info("📊 Sem máscara de perda - todos os pixels serão usados")
        
        self.optimizer = self._create_optimizer()
    
    def _compute_pos_weight(self) -> float:
        """Calcula peso para classe positiva com limites conservadores."""
        total = 0
        positives = 0
        
        max_samples = min(1000, len(self.train_dataset))
        for i in range(max_samples):
            batch = self.train_dataset[i]
            y = batch["y_bin"]
            # ✅ Usar a máscara para contar apenas pixels válidos
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
        
        # Abordagem mais suave: usar raiz quarta para evitar overconfidence
        pos_weight = raw_pos_weight ** 0.25
        
        # Limite máximo conservador
        max_pos_weight = getattr(self.config.imbalance, 'max_pos_weight', 3.0)
        pos_weight = np.clip(pos_weight, 1.0, max_pos_weight)
        
        self.logger.debug(
            f"Pos_weight adaptativo: {pos_weight:.3f} | "
            f"Prevalência real: {prevalence:.2%} | "
            f"Raw: {raw_pos_weight:.1f}"
        )
        
        return pos_weight
    
    def _create_optimizer(self):
        """Cria otimizador com grupos separados para transfer learning."""
        if not self.freeze_encoder_first_epoch:
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        
        encoder_params = []
        decoder_params = []
        
        for name, param in self.model.named_parameters():
            if name.startswith('encoder') or name.startswith('attention'):
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        
        encoder_lr = self.config.training.learning_rate * self.encoder_lr_factor
        
        return torch.optim.Adam([
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': decoder_params, 'lr': self.config.training.learning_rate}
        ], weight_decay=self.config.training.weight_decay)
    
    def train_epoch(self, train_loader):
        """Treina uma época."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        
        for batch in train_loader:
            x = batch["x"].to(self.device)
            y_bin = batch["y_bin"].to(self.device).float().unsqueeze(1)
            
            # ✅ CORREÇÃO: Obter máscara do batch
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(self.device).unsqueeze(1)  # (B, 1, H, W)
            
            self.optimizer.zero_grad(set_to_none=True)
            logits, _ = self.model(x)
            
            # Calcular perda
            loss_raw = self.criterion(logits, y_bin)  # (B, 1, H, W)
            
            # ✅ CORREÇÃO: Aplicar máscara à perda
            if mask is not None:
                # Máscara: True para pixels válidos, False para inválidos
                masked_loss = loss_raw * mask
                valid_pixels = mask.sum()
                if valid_pixels > 0:
                    loss = masked_loss.sum() / valid_pixels
                else:
                    loss = loss_raw.mean()  # Fallback
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
        """Coleta probabilidades e targets, aplicando máscara."""
        self.model.eval()
        all_probs = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch["x"].to(self.device)
                y_bin = batch["y_bin"].to(self.device).float().unsqueeze(1)
                
                # ✅ Obter máscara
                mask = batch.get("mask", None)
                if mask is not None:
                    all_masks.append(mask.cpu().numpy().flatten())
                
                logits, _ = self.model(x)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
                all_probs.extend(probs)
                all_targets.extend(y_bin.cpu().numpy().flatten())
        
        probs = np.array(all_probs)
        targets = np.array(all_targets)
        
        # ✅ CORREÇÃO: Aplicar máscara para filtrar
        if all_masks:
            masks = np.concatenate(all_masks)
            valid_idx = masks > 0.5
            probs = probs[valid_idx]
            targets = targets[valid_idx]
        
        # Remover NaNs
        valid = ~(np.isnan(probs) | np.isnan(targets))
        probs = probs[valid]
        targets = targets[valid]
        
        return probs, targets

    def fit_calibrator(self, train_loader, val_loader):
        from sklearn.linear_model import LogisticRegression
        
        # Coletar dados
        train_probs, train_targets = self._collect_predictions(train_loader)
        val_probs, val_targets = self._collect_predictions(val_loader)
        
        probs = np.concatenate([train_probs, val_probs])
        targets = np.concatenate([train_targets, val_targets])
        
        # ✅ Platt Scaling com reshape
        self.calibrator = LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.calibrator.fit(probs.reshape(-1, 1), targets)  # ✅ CORRETO
        
        return self.calibrator
    
    def calibrate_probs(self, probs: np.ndarray) -> np.ndarray:
        """Aplica calibração nas probabilidades."""
        if self.calibrator is None:
            return probs
        
        from sklearn.linear_model import LogisticRegression
        if isinstance(self.calibrator, LogisticRegression):
            return self.calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
        return self.calibrator.predict(probs)
    
    def validate(self, val_loader, use_calibration: bool = True):
        """Valida o modelo e otimiza threshold."""
        probs, targets_np = self._collect_predictions(val_loader)
        
        # Aplicar calibração se disponível e solicitado
        if use_calibration and self.calibrator is not None:
            probs_original = probs.copy()
            probs = self.calibrate_probs(probs)
            
            self.logger.debug(f"   Calibração aplicada: {probs_original.mean():.3f} → {probs.mean():.3f}")
        
        if len(probs) == 0:
            return {"threshold": 0.14, "metrics": {"csi": 0.0, "mcc": 0.0}, "csi": 0.0, "mcc": 0.0}
        
        # Log detalhado da distribuição
        self.logger.debug(f"   Prob stats: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
        self.logger.debug(f"   Percentis: 1%={np.percentile(probs, 1):.4f}, 5%={np.percentile(probs, 5):.4f}")
        self.logger.debug(f"              10%={np.percentile(probs, 10):.4f}, 50%={np.percentile(probs, 50):.4f}")
        self.logger.debug(f"              90%={np.percentile(probs, 90):.4f}, 95%={np.percentile(probs, 95):.4f}, 99%={np.percentile(probs, 99):.4f}")
        self.logger.debug(f"   Target positive rate: {targets_np.mean():.4f}")
        
        # Usar None para range adaptativo automático
        best_thr, best_metrics = find_best_threshold(
            probs, targets_np, 
            thresholds=None,
            metric=self.optimization_metric,
        )
        
        return {
            "threshold": best_thr,
            "metrics": best_metrics,
            "csi": best_metrics["csi"],
            "mcc": best_metrics["mcc"],
            "score": best_metrics[self.optimization_metric],
        }
    
    def train(self):
        """Loop principal de treinamento."""
        self.logger.header("TREINAMENTO DO PREDICTOR")
        self.logger.info(f"Métrica de otimização: {self.optimization_metric.upper()}")
        
        train_loader, val_loader = self.create_dataloaders(shuffle_train=True)
        
        best_score = -1.0
        best_csi = -1.0
        patience_counter = 0
        min_epochs = 10
        
        if self.freeze_encoder_first_epoch:
            self.model.freeze_encoder(freeze=True)
            self.logger.info(f"🔒 Encoder + Attention congelados por {self.freeze_encoder_epochs} épocas")
        
        for epoch in range(self.config.training.epochs):
            if self.freeze_encoder_first_epoch and epoch == self.freeze_encoder_epochs:
                self.model.freeze_encoder(freeze=False)
                new_lr = self.config.training.learning_rate * 0.5
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=new_lr,
                    weight_decay=self.config.training.weight_decay,
                )
                self.logger.info(f"🔓 Encoder + Attention descongelados na época {epoch} (LR={new_lr:.2e})")
            
            train_loss = self.train_epoch(train_loader)
            
            # Validação SEM calibração durante o treinamento
            val_results = self.validate(val_loader, use_calibration=False)
            
            epoch_score = val_results["score"]
            epoch_csi = val_results["csi"]
            epoch_mcc = val_results["mcc"]
            
            # Salva checkpoint baseado no melhor CSI
            improved = epoch_csi > best_csi
            if improved:
                best_score = epoch_score
                best_csi = epoch_csi
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
            
            if improved:
                status = f"{Colors.GREEN}✓{Colors.RESET}"
            else:
                status = f"ES {patience_counter}/{self.config.training.patience}"
            
            self.logger.info(
                f"Época {epoch:3d} | Loss {train_loss:.4f} | "
                f"CSI {epoch_csi:.4f} | MCC {epoch_mcc:.4f} | "
                f"Thr {val_results['threshold']:.3f} | "
                f"FAR {val_results['metrics'].get('far', 0.0):.3f} | {status}"
            )
            
            if self.freeze_encoder_first_epoch and epoch < self.freeze_encoder_epochs:
                self.logger.debug(f"   🔒 Encoder congelado (época {epoch+1}/{self.freeze_encoder_epochs})")
            
            if epoch >= min_epochs and patience_counter >= self.config.training.patience:
                self.logger.warning(f"Early stopping na época {epoch}")
                break
        
        # --- Carrega o melhor checkpoint ---
        checkpoint = torch.load(self.save_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # ✅ Usar TREINO + VALIDAÇÃO com Platt Scaling
        if self.calibrate:
            self.fit_calibrator(train_loader, val_loader)
        
        # --- AVALIAÇÃO FINAL COM CALIBRAÇÃO ---
        if self.calibrate and self.calibrator is not None:
            self.logger.info("📊 Avaliação final com calibração...")
            final_val_calibrated = self.validate(val_loader, use_calibration=True)
            final_val = final_val_calibrated
            
            # Também avalia sem calibração para comparação
            final_val_standard = self.validate(val_loader, use_calibration=False)
            
            self.logger.info("📊 Comparação:")
            self.logger.info(f"   Sem calibração: CSI={final_val_standard['csi']:.4f}, Thr={final_val_standard['threshold']:.3f}")
            self.logger.info(f"   Com calibração:  CSI={final_val_calibrated['csi']:.4f}, Thr={final_val_calibrated['threshold']:.3f}")
            
            # Salva ambos no checkpoint
            checkpoint["final_metrics_uncalibrated"] = final_val_standard["metrics"]
            checkpoint["final_metrics"] = final_val_calibrated["metrics"]
            checkpoint["threshold_uncalibrated"] = final_val_standard["threshold"]
            checkpoint["best_threshold"] = final_val_calibrated["threshold"]
            checkpoint["best_csi"] = final_val_calibrated["csi"]
            checkpoint["best_mcc"] = final_val_calibrated["mcc"]
            checkpoint["calibrated"] = True
            
            # --- SALVAR CALIBRADOR ---
            try:
                import joblib
                
                # Salvar calibrador como arquivo pickle separado
                calib_path = self.save_path.parent / "calibrator.pkl"
                joblib.dump(self.calibrator, calib_path)
                self.logger.success(f"✅ Calibrador salvo em: {calib_path}")
                
                # Também salvar no checkpoint para compatibilidade
                checkpoint["calibrator"] = self.calibrator
                
                # Salvar calibrador no diretório de grid search também
                gs_calib_path = self.save_path.parent.parent / "calibrator.pkl"
                joblib.dump(self.calibrator, gs_calib_path)
                self.logger.debug(f"   Calibrador também salvo em: {gs_calib_path}")
                
            except ImportError:
                self.logger.warning("⚠️ joblib não disponível. Calibrador não salvo como pickle.")
                checkpoint["calibrator"] = self.calibrator
                
            except Exception as e:
                self.logger.warning(f"⚠️ Erro ao salvar calibrador: {e}")
                checkpoint["calibrator"] = self.calibrator
        
        else:
            # Sem calibração
            final_val = self.validate(val_loader, use_calibration=False)
            checkpoint["best_threshold"] = final_val["threshold"]
            checkpoint["best_score"] = final_val["score"]
            checkpoint["best_csi"] = final_val["csi"]
            checkpoint["best_mcc"] = final_val["mcc"]
            checkpoint["best_far"] = final_val["metrics"].get("far", 0.0)
            checkpoint["final_metrics"] = final_val["metrics"]
            checkpoint["calibrated"] = False
        
        # Salvar checkpoint atualizado
        torch.save(checkpoint, self.save_path)
        
        self.logger.success(
            f"Final - CSI: {final_val['csi']:.4f} | MCC: {final_val['mcc']:.4f} | "
            f"FAR: {final_val['metrics'].get('far', 0.0):.4f} | "
            f"Threshold: {final_val['threshold']:.3f}"
        )
        
        return best_score

    def predict_with_calibration(self, logits: torch.Tensor) -> np.ndarray:
        """Faz previsão com calibração."""
        probs = torch.sigmoid(logits).cpu().numpy()
        if self.calibrator is not None:
            probs = self.calibrate_probs(probs.flatten()).reshape(probs.shape)
        return probs