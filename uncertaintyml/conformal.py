"""
Conformal Prediction wrapper for rigorous uncertainty quantification.
Uses MAPIE (Model Agnostic Prediction Interval Estimator).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any

try:
    from mapie.classification import MapieClassifier
    from mapie.metrics import classification_coverage_score, classification_mean_width_score
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False


class ConformalCalibrator:
    """
    Wraps a base classifier to provide statistically guaranteed prediction sets.
    """

    def __init__(self, estimator: Any, method: str = "score", cv: str = "prefit"):
        """
        Args:
            estimator: Trained scikit-learn compatible classifier.
            method: Conformal method ('score', 'lac', 'aps').
            cv: 'prefit' assumes estimator is already trained.
        """
        if not MAPIE_AVAILABLE:
            print("⚠️ MAPIE not installed. Conformal prediction disabled.")
            self.mapie = None
            return

        self.mapie = MapieClassifier(estimator=estimator, method=method, cv=cv)
        self.calibrated = False

    def fit(self, X_cal: pd.DataFrame, y_cal: pd.Series) -> None:
        """Calibrate using a hold-out dataset."""
        if not self.mapie:
            return
        
        # MAPIE expects numpy arrays
        X_np = X_cal.values if hasattr(X_cal, "values") else X_cal
        y_np = y_cal.values if hasattr(y_cal, "values") else y_cal
        
        self.mapie.fit(X_np, y_np)
        self.calibrated = True

    def predict(self, X: pd.DataFrame, alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict sets with 1-alpha confidence (e.g., 90% for alpha=0.1).
        
        Returns:
            y_pred: Point predictions
            y_sets: Boolean array (n_samples, n_classes) where True means class is in set.
        """
        if not self.mapie or not self.calibrated:
            # Fallback: return point prediction and empty sets
            if hasattr(self.mapie, "estimator"):
                return self.mapie.estimator.predict(X), np.zeros((len(X), 2), dtype=bool)
            return np.zeros(len(X)), np.zeros((len(X), 2), dtype=bool)

        X_np = X.values if hasattr(X, "values") else X
        y_pred, y_sets = self.mapie.predict(X_np, alpha=alpha)
        
        # y_sets is shape (n_samples, n_classes, n_alphas). Squeeze if single alpha.
        if y_sets.ndim == 3 and y_sets.shape[2] == 1:
            y_sets = y_sets[:, :, 0]
            
        return y_pred, y_sets

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, alpha: float = 0.1) -> Dict[str, float]:
        """Calculate coverage and average set size."""
        if not self.mapie or not self.calibrated:
            return {}

        _, y_sets = self.predict(X_test, alpha=alpha)
        y_true = y_test.values if hasattr(y_test, "values") else y_test
        
        coverage = classification_coverage_score(y_true, y_sets)
        width = classification_mean_width_score(y_sets)
        
        return {
            "coverage": round(coverage, 4),
            "avg_set_size": round(width, 4),
            "target_coverage": 1 - alpha
        }
