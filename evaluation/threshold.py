"""threshold_optimizer.py - Recalibração de threshold para teste"""

import torch
import numpy as np
import json
from pathlib import Path

from config import ExperimentConfig, get_paths
from data import load_region_timeseries, ClimateNormalizer, load_spi_cache
from models import ConvLSTMPredictor
from evaluation.metrics import find_best_threshold


class ThresholdOptimizer:
    """Otimizador de threshold para dados de teste."""
    
    def __init__(self, config: ExperimentConfig, base_data_path: Path):
        self.config = config
        self.base_data_path = base_data_path
        self.threshold = config.spi.threshold
        self.paths = get_paths(config.region, threshold=self.threshold)
    
    def load_best_config(self) -> dict:
        """Carrega melhor configuração do grid search."""
        candidates = [
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
            raise FileNotFoundError(f"Configuração não encontrada em: {self.paths['grid_search_results']}")
        
        with open(best_path, "r") as f:
            data = json.load(f)
        
        if "best_configuration" in data:
            return data["best_configuration"]
        return data
    
    def load_model(self, best_config: dict) -> ConvLSTMPredictor:
        """Carrega o melhor modelo."""
        p = best_config["p"]
        q = best_config["q"]
        use_tl = best_config.get("transfer_learning", True)
        
        model_type = "pretrained" if use_tl else "scratch"
        model_dir = self.paths["grid_search_pretrained"] if use_tl else self.paths["grid_search_scratch"]
        checkpoint_path = model_dir / f"model_p{p}_q{q}.pth"
        
        if not checkpoint_path.exists():
            checkpoint_path = self.paths["grid_search_dir"] / model_type / f"model_p{p}_q{q}.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
        
        print(f"📦 Carregando modelo: {checkpoint_path}")
        
        model = ConvLSTMPredictor(self.config.get_model_config("predictor")).to(self.config.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        
        return model
    
    def get_test_indices(self, data_len: int, p: int, q: int) -> list:
        """Retorna índices para dados de teste."""
        min_required = p + q
        if data_len <= min_required:
            return []
        return list(range(p, data_len - q))
    
    def optimize(self, use_validation_fallback: bool = True):
        """Executa otimização de threshold."""
        print("\n" + "=" * 70)
        print("🔧 OTIMIZADOR DE THRESHOLD")
        print("=" * 70)
        
        best_config = self.load_best_config()
        model = self.load_model(best_config)
        
        p = best_config["p"]
        q = best_config["q"]
        
        print(f"  Região: {self.config.region}")
        print(f"  SPI Threshold: {self.config.spi.threshold}")
        print(f"  p={p}, q={q}")
        
        # Carregar dados
        out = load_region_timeseries(self.base_data_path, self.config)
        spi, _ = load_spi_cache(self.config.spi.scale, self.paths["spi_cache_dir"])
        
        # Split temporal
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
            print(f"\n✅ Teste OK: {test_len} meses")
            eval_mask = test_mask
            split_name = "test"
        elif use_validation_fallback:
            val_len = val_mask.sum()
            print(f"\n⚠️ Teste curto ({test_len} < {min_required})")
            print(f"   Usando validação ({val_len} meses) para calibração...")
            eval_mask = val_mask
            split_name = "validation"
        else:
            raise RuntimeError(f"Dados de teste insuficientes ({test_len} < {min_required})")
        
        data_eval = out["data"][eval_mask]
        months_eval = months[eval_mask]
        spi_eval = spi[eval_mask] if spi is not None else None
        valid_mask = out["valid_mask"]
        
        # Carregar normalizador
        normalizer_path = self.paths["autoencoder_dir"] / "normalizer.json"
        if not normalizer_path.exists():
            normalizer_path = self.paths["grid_search_dir"] / "normalizer.json"
        
        if not normalizer_path.exists():
            raise FileNotFoundError(f"Normalizador não encontrado: {normalizer_path}")
        
        normalizer = ClimateNormalizer.load(normalizer_path)
        print(f"✅ Normalizador carregado de: {normalizer_path}")
        
        # Normalizar
        data_norm = normalizer.transform(data_eval, months=months_eval, valid_mask=valid_mask)
        data_norm = np.nan_to_num(data_norm, nan=0.0)
        
        # Formatar dados
        data_ch = np.transpose(data_norm, (0, 3, 1, 2))
        
        indices = self.get_test_indices(len(data_ch), p, q)
        
        if not indices:
            print("⚠️ Nenhuma amostra disponível!")
            return None, None
        
        # Binary mask
        binary_mask = (spi_eval <= self.config.spi.threshold).astype(np.float32)
        # ✅ NÃO aplicar máscara ao binary_mask
        # if valid_mask is not None:
        #     binary_mask[:, ~valid_mask] = 0.0  # ❌ REMOVIDO
        
        print(f"\n🔮 Coletando predições em {len(indices)} amostras ({split_name})...")
        
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for t in indices:
                target_idx = t + q
                x_seq = data_ch[t - p:t]
                x_tensor = torch.from_numpy(x_seq).float().unsqueeze(0).to(self.config.device)
                
                logits, _ = model(x_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
                
                # ✅ NÃO aplicar máscara aos dados
                # if valid_mask is not None:
                #     probs = probs * valid_mask  # ❌ REMOVIDO
                
                target = binary_mask[target_idx]
                
                all_probs.append(probs.flatten())
                all_targets.append(target.flatten())
        
        probs = np.concatenate(all_probs)
        targets = np.concatenate(all_targets)
        
        # ✅ Usar máscara apenas para FILTRAR
        if valid_mask is not None:
            mask_flat = valid_mask.flatten()
            n_pixels = len(mask_flat)
            n_total = len(probs)
            # Expandir máscara para corresponder ao número de pixels
            mask_expanded = np.tile(mask_flat, n_total // n_pixels + 1)[:n_total]
            valid_idx = mask_expanded > 0
            probs = probs[valid_idx]
            targets = targets[valid_idx]
        
        valid = ~(np.isnan(probs) | np.isnan(targets))
        probs = probs[valid]
        targets = targets[valid]
        
        n_pos = (targets == 1).sum()
        print(f"   Total pixels válidos: {len(probs):,}")
        print(f"   Amostras positivas: {n_pos:,} ({100*n_pos/len(targets):.4f}%)")
        
        # Otimizar threshold
        thresholds = np.arange(0.01, 0.99, 0.005)
        best_thr, best_metrics = find_best_threshold(probs, targets, thresholds)
        
        print(f"\n✅ Threshold otimizado ({split_name}): {best_thr:.3f}")
        print(f"   CSI: {best_metrics['csi']:.4f}")
        print(f"   MCC: {best_metrics['mcc']:.4f}")
        print(f"   Precision: {best_metrics['precision']:.4f}")
        print(f"   Recall: {best_metrics['recall']:.4f}")
        print(f"   F1: {best_metrics['f1']:.4f}")
        
        # Salvar resultado
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
        
        print(f"\n📁 Resultado salvo em: {output_path}")
        print("=" * 70)
        
        return best_thr, best_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Otimizador de threshold para teste")
    parser.add_argument("--region", type=str, default="Sul", help="Região")
    parser.add_argument("--threshold-spi", type=float, default=-2.0, help="Threshold SPI")
    parser.add_argument("--no-fallback", action="store_true", help="Não usar validação como fallback")
    
    args = parser.parse_args()
    
    config = ExperimentConfig()
    config.region = args.region
    config.spi.threshold = args.threshold_spi
    
    from config.paths import get_data_path
    base_data_path = get_data_path()
    
    optimizer = ThresholdOptimizer(config, base_data_path)
    optimizer.optimize(use_validation_fallback=not args.no_fallback)