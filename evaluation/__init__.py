"""Evaluation package - Metrics, threshold optimization and calibration."""

from .metrics import (
    compute_confusion_matrix,
    compute_metrics,
    aggregate_metrics,
    find_best_threshold,
)
from .threshold import ThresholdOptimizer
from .calibration import PlattCalibrator, IsotonicCalibrator

__all__ = [
    "compute_confusion_matrix",
    "compute_metrics",
    "aggregate_metrics",
    "find_best_threshold",
    "ThresholdOptimizer",
    "PlattCalibrator",
    "IsotonicCalibrator",
]