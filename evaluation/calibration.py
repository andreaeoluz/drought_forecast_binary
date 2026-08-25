"""calibration.py - Probability calibration methods."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union


class PlattCalibrator:
    """Platt Scaling (Logistic Regression) for probability calibration."""

    def __init__(self, C: float = 0.1, max_iter: int = 1000, random_state: int = 42):
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight='balanced',
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, probs: np.ndarray, targets: np.ndarray) -> "PlattCalibrator":
        """Fit the calibrator."""
        self.model.fit(probs.reshape(-1, 1), targets)
        self._fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities."""
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted.")
        original_shape = probs.shape
        return self.model.predict_proba(probs.reshape(-1, 1))[:, 1].reshape(original_shape)


class IsotonicCalibrator:
    """Isotonic Regression for probability calibration."""

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')
        self._fitted = False

    def fit(self, probs: np.ndarray, targets: np.ndarray) -> "IsotonicCalibrator":
        """Fit the calibrator."""
        self.model.fit(probs, targets)
        self._fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities."""
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted.")
        return self.model.predict(probs)