"""metrics.py - Binary classification metrics."""

import math
from typing import Dict, Optional, List, Tuple
import numpy as np


def compute_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
) -> Tuple[int, int, int, int]:
    """
    Compute the confusion matrix.

    Args:
        preds: Predictions array (0/1).
        targets: Targets array (0/1).

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
    """Compute classification metrics from a confusion matrix."""
    eps = 1e-7

    tp_f = float(tp)
    fp_f = float(fp)
    fn_f = float(fn)
    tn_f = float(tn)

    csi = tp_f / (tp_f + fp_f + fn_f + eps)
    precision = tp_f / (tp_f + fp_f + eps)
    recall = tp_f / (tp_f + fn_f + eps)
    far = fp_f / (tp_f + fp_f + eps)
    bias = (tp_f + fp_f) / (tp_f + fn_f + eps)

    # --- MCC, guarded against degenerate confusion matrices ---
    total = tp_f + fp_f + fn_f + tn_f

    if total < eps:
        mcc = 0.0
    else:
        numerator = (tp_f * tn_f) - (fp_f * fn_f)

        denom_tp_fp = max(tp_f + fp_f, 0.0)
        denom_tp_fn = max(tp_f + fn_f, 0.0)
        denom_tn_fp = max(tn_f + fp_f, 0.0)
        denom_tn_fn = max(tn_f + fn_f, 0.0)

        product = denom_tp_fp * denom_tp_fn * denom_tn_fp * denom_tn_fn

        if product < eps:
            # Not enough information to compute a meaningful MCC.
            mcc = 0.0
        else:
            denominator = math.sqrt(product)
            mcc = numerator / denominator
            mcc = max(-1.0, min(1.0, mcc))

    accuracy = (tp_f + tn_f) / (total + eps)

    tpr = recall
    tnr = tn_f / (tn_f + fp_f + eps)
    balanced_accuracy = (tpr + tnr) / 2

    f1 = 2 * (precision * recall) / (precision + recall + eps)

    informedness = tpr + tnr - 1

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


def compute_metrics_from_arrays(
    preds: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """Compute metrics directly from prediction and target arrays."""
    tp, fp, fn, tn = compute_confusion_matrix(preds, targets)
    return compute_metrics(tp, fp, fn, tn)


def aggregate_metrics(confusion_list: List[Tuple[int, int, int, int]]) -> Dict[str, float]:
    """Aggregate multiple confusion matrices into a single set of metrics."""
    tp = sum(c[0] for c in confusion_list)
    fp = sum(c[1] for c in confusion_list)
    fn = sum(c[2] for c in confusion_list)
    tn = sum(c[3] for c in confusion_list)

    return compute_metrics(tp, fp, fn, tn)


def find_best_threshold(probs, targets, thresholds=None, metric='mcc'):
    """
    Find the decision threshold that maximizes the given metric.

    Args:
        probs: Predicted probabilities.
        targets: Binary targets.
        thresholds: Candidate thresholds. If None, an adaptive range is
            derived from the probability distribution.
        metric: 'mcc', 'csi', or any other key returned by compute_metrics.

    Returns:
        (best_threshold, best_metrics)
    """
    if thresholds is None:
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
        metrics = compute_metrics_from_arrays(preds, targets)

        if metric == 'mcc':
            score = metrics.get('mcc', 0.0)
        elif metric == 'csi':
            score = metrics.get('csi', 0.0)
        else:
            score = metrics.get(metric, 0.0)

        if score > best_score:
            best_score = score
            best_thr = thr
            best_metrics = metrics
        elif abs(score - best_score) < 1e-6:
            # Break ties by preferring the higher CSI.
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
    Compute metrics after spatial post-processing (small-object removal).

    Args:
        probs: Probabilities (H, W).
        targets: Binary targets (H, W).
        threshold: Binarization threshold.
        min_area: Minimum area to keep a connected component.
        valid_mask: Optional validity mask.

    Returns:
        Dict of computed metrics.
    """
    from utils.spatial import postprocess_binary_mask

    # Copy before mutating in place, since `targets` may be the caller's array.
    targets = targets.copy()

    # Same boundary convention as find_best_threshold (probs >= thr), so
    # metrics computed with vs. without post-processing at the same
    # threshold value are directly comparable pixel-for-pixel.
    binary = (probs >= threshold).astype(np.uint8)

    if binary.sum() > 0:
        binary = postprocess_binary_mask(
            binary.astype(np.float32),
            threshold=0.5,
            min_area=min_area,
            hole_area=max(1, min_area // 2),
        )
        binary = binary.astype(np.uint8)

    if valid_mask is not None:
        binary[~valid_mask] = 0
        targets[~valid_mask] = 0

    return compute_metrics_from_arrays(binary, targets)