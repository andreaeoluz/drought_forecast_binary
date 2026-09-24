"""Evaluation package - Metrics, threshold optimization and calibration."""

from .metrics import (
    compute_confusion_matrix,
    compute_metrics,
    compute_metrics_from_arrays,
    aggregate_metrics,
    find_best_threshold,
    compute_metrics_with_postprocessing,
)
from .threshold import ThresholdOptimizer
from .calibration import PlattCalibrator, IsotonicCalibrator

__all__ = [
    "compute_confusion_matrix",
    "compute_metrics",
    "compute_metrics_from_arrays",
    "aggregate_metrics",
    "find_best_threshold",
    "compute_metrics_with_postprocessing",
    "ThresholdOptimizer",
    "PlattCalibrator",
    "IsotonicCalibrator",
]