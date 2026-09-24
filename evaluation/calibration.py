"""calibration.py - Probability calibration methods."""

from typing import Optional, Union

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Platt Scaling (Logistic Regression) for probability calibration."""

    def __init__(
        self,
        C: float = 0.1,
        max_iter: int = 1000,
        random_state: int = 42,
        class_weight: Optional[Union[str, dict]] = None,
    ):
        # class_weight defaults to None (unweighted): Platt scaling is meant
        # to learn the true mapping from raw score to empirical frequency.
        # Balancing classes here would fit the calibrator to an artificial
        # 50/50 distribution instead of the real (often ~2-15%) drought
        # prevalence, actively working against calibration's own purpose.
        # Pass class_weight='balanced' explicitly if that's ever needed.
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
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