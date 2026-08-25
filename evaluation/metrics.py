"""metrics.py - Métricas para classificação binária"""

import math
from typing import Dict, Optional, List, Tuple
import numpy as np
import torch


def compute_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
) -> Tuple[int, int, int, int]:
    """
    Calcula matriz de confusão.
    
    Args:
        preds: Array de previsões (0/1)
        targets: Array de alvos (0/1)
    
    Returns:
        (tp, fp, fn, tn)
    """
    tp = np.sum((preds == 1) & (targets == 1))
    fp = np.sum((preds == 1) & (targets == 0))
    fn = np.sum((preds == 0) & (targets == 1))
    tn = np.sum((preds == 0) & (targets == 0))
    
    return int(tp), int(fp), int(fn), int(tn)


def compute_metrics(
    tp: int, 
    fp: int, 
    fn: int, 
    tn: int,
) -> Dict[str, float]:
    """
    Calcula métricas a partir da matriz de confusão.
    """
    eps = 1e-7
    
    # Converter para float
    tp_f = float(tp)
    fp_f = float(fp)
    fn_f = float(fn)
    tn_f = float(tn)
    
    # CSI
    csi = tp_f / (tp_f + fp_f + fn_f + eps)
    
    # Precision
    precision = tp_f / (tp_f + fp_f + eps)
    
    # Recall
    recall = tp_f / (tp_f + fn_f + eps)
    
    # FAR
    far = fp_f / (tp_f + fp_f + eps)
    
    # Bias
    bias = (tp_f + fp_f) / (tp_f + fn_f + eps)
    
    # --- MCC ROBUSTO ---
    total = tp_f + fp_f + fn_f + tn_f
    
    # Se total for zero, retornar 0
    if total < eps:
        mcc = 0.0
    else:
        # Calcular numerator
        numerator = (tp_f * tn_f) - (fp_f * fn_f)
        
        # Calcular denominador com proteção
        denom_tp_fp = max(tp_f + fp_f, 0.0)
        denom_tp_fn = max(tp_f + fn_f, 0.0)
        denom_tn_fp = max(tn_f + fp_f, 0.0)
        denom_tn_fn = max(tn_f + fn_f, 0.0)
        
        product = denom_tp_fp * denom_tp_fn * denom_tn_fp * denom_tn_fn
        
        # Se produto for zero, MCC = 0 (não há informação suficiente)
        if product < eps:
            mcc = 0.0
        else:
            denominator = math.sqrt(product)
            mcc = numerator / denominator
            
            # Garantir que MCC esteja no intervalo [-1, 1]
            mcc = max(-1.0, min(1.0, mcc))
    
    # Accuracy
    accuracy = (tp_f + tn_f) / (total + eps)
    
    # TPR e TNR
    tpr = recall
    tnr = tn_f / (tn_f + fp_f + eps)
    
    # Balanced Accuracy
    balanced_accuracy = (tpr + tnr) / 2
    
    # F1
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    
    # Informedness
    informedness = tpr + tnr - 1
    
    # Markedness
    ppv = precision
    npv = tn_f / (tn_f + fn_f + eps)
    markedness = ppv + npv - 1
    
    return {
        "csi": round(csi, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "far": round(far, 6),
        "bias": round(bias, 6),
        "mcc": round(mcc, 6),
        "accuracy": round(accuracy, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "f1": round(f1, 6),
        "informedness": round(informedness, 6),
        "markedness": round(markedness, 6),
        "specificity": round(tnr, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ✅ FUNÇÃO AUXILIAR: calcular métricas a partir de preds e targets
def compute_metrics_from_arrays(
    preds: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Calcula métricas diretamente de arrays de predições e targets.
    
    Args:
        preds: Array de previsões (0/1)
        targets: Array de alvos (0/1)
    
    Returns:
        Dicionário com métricas
    """
    tp, fp, fn, tn = compute_confusion_matrix(preds, targets)
    return compute_metrics(tp, fp, fn, tn)


def aggregate_metrics(confusion_list: List[Tuple[int, int, int, int]]) -> Dict[str, float]:
    """Agrega múltiplas matrizes de confusão."""
    tp = sum(c[0] for c in confusion_list)
    fp = sum(c[1] for c in confusion_list)
    fn = sum(c[2] for c in confusion_list)
    tn = sum(c[3] for c in confusion_list)
    
    return compute_metrics(tp, fp, fn, tn)


def find_best_threshold(probs, targets, thresholds=None, metric='mcc'):
    """
    Encontra o melhor threshold baseado na métrica especificada.
    
    Args:
        metric: 'mcc', 'csi', 'f1', etc.
    """
    if thresholds is None:
        # Range adaptativo
        p1 = np.percentile(probs, 1)
        p99 = np.percentile(probs, 99)
        low = max(0.001, p1 - 0.02)
        high = min(0.999, p99 + 0.02)
        thresholds = np.arange(low, high + 0.005, 0.005)
    
    best_score = -1.0
    best_thr = thresholds[0]
    best_metrics = None
    
    for thr in thresholds:
        preds = (probs >= thr).astype(np.int32)
        
        # ✅ USAR compute_metrics_from_arrays para obter métricas
        metrics = compute_metrics_from_arrays(preds, targets)
        
        # ✅ Priorizar MCC para seleção
        if metric == 'mcc':
            score = metrics.get('mcc', 0.0)
        elif metric == 'csi':
            score = metrics.get('csi', 0.0)
        else:
            score = metrics.get(metric, 0.0)
        
        # Desempate: se MCC igual, usar CSI
        if score > best_score:
            best_score = score
            best_thr = thr
            best_metrics = metrics
        elif abs(score - best_score) < 1e-6:
            # Desempate por CSI
            current_csi = metrics.get('csi', 0.0)
            best_csi = best_metrics.get('csi', 0.0) if best_metrics else 0.0
            if current_csi > best_csi:
                best_thr = thr
                best_metrics = metrics
    
    return best_thr, best_metrics


def compute_metrics_with_postprocessing(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    min_area: int = 5,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict:
    """
    Calcula métricas com pós-processamento espacial.
    
    Aplica remoção de pequenos objetos antes de calcular as métricas.
    
    Args:
        probs: Probabilidades (H, W)
        targets: Targets binários (H, W)
        threshold: Limiar de binarização
        min_area: Área mínima para manter um componente
        valid_mask: Máscara de pixels válidos (opcional)
    
    Returns:
        Dicionário com métricas calculadas
    """
    from utils.spatial import postprocess_binary_mask
    
    # Binarizar
    binary = (probs > threshold).astype(np.uint8)
    
    # Aplicar pós-processamento
    if binary.sum() > 0:
        binary = postprocess_binary_mask(
            binary.astype(np.float32),
            threshold=0.5,
            min_area=min_area,
            hole_area=max(1, min_area // 2),
        )
        binary = binary.astype(np.uint8)
    
    # Aplicar máscara de validade
    if valid_mask is not None:
        binary[~valid_mask] = 0
        targets[~valid_mask] = 0
    
    # ✅ USAR compute_metrics_from_arrays
    return compute_metrics_from_arrays(binary, targets)