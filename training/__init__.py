"""Training package - Training logic for models."""

from .base import BaseTrainer
from .autoencoder import AutoencoderTrainer
from .predictor import PredictorTrainer
from .losses import FocalLoss, WeightedSmoothL1Loss, WeightedBCEWithLogitsLoss, build_loss
from .rolling_origin import (
    RollingSplit,
    build_rolling_origin_splits,
    aggregate_rolling_metrics,
    run_rolling_origin_evaluation,
)

__all__ = [
    "BaseTrainer",
    "AutoencoderTrainer",
    "PredictorTrainer",
    "FocalLoss",
    "WeightedSmoothL1Loss",
    "WeightedBCEWithLogitsLoss",
    "build_loss",
    "RollingSplit",
    "build_rolling_origin_splits",
    "aggregate_rolling_metrics",
    "run_rolling_origin_evaluation",
]