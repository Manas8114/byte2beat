"""
Evaluation module: K-Fold CV, calibration curves, ECE, bootstrap CIs, Brier score.

Moved from src/evaluation.py with identical logic.
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from typing import Callable, Dict, List, Optional


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error. Lower is better (0 = perfect).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = y_prob[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)
    return ece


def compute_calibration_data(y_true, y_prob, n_bins: int = 10) -> Dict:
    """Calibration curve data for visualization."""
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    ece = compute_ece(y_true, y_prob, n_bins)
    return {
        "fraction_of_positives": fraction_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
        "ece": float(ece),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Dict:
    """Bootstrap confidence interval for any metric function."""
    n = len(y_true)
    scores = []

    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, n)
        y_true_sample = y_true[indices]
        y_prob_sample = y_prob[indices]
        if len(np.unique(y_true_sample)) < 2:
            continue
        scores.append(metric_fn(y_true_sample, y_prob_sample))

    scores = np.array(scores)
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(scores)),
        "lower": float(np.percentile(scores, alpha * 100)),
        "upper": float(np.percentile(scores, (1 - alpha) * 100)),
        "std": float(np.std(scores)),
    }


def cross_validate_model(model_class, X, y, n_splits: int = 5, **model_kwargs) -> Dict:
    """Stratified K-Fold cross-validation with AUC, accuracy, and ECE."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_aucs, fold_accs, fold_eces = [], [], []
    all_probs, all_labels = [], []

    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
        else:
            y_prob = model.predict(X_val)

        fold_aucs.append(roc_auc_score(y_val, y_prob))
        fold_accs.append(accuracy_score(y_val, (y_prob > 0.5).astype(int)))
        fold_eces.append(compute_ece(y_val, y_prob))
        all_probs.extend(y_prob)
        all_labels.extend(y_val)

    return {
        "n_splits": n_splits,
        "fold_aucs": fold_aucs,
        "fold_accs": fold_accs,
        "fold_eces": fold_eces,
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "mean_acc": float(np.mean(fold_accs)),
        "std_acc": float(np.std(fold_accs)),
        "mean_ece": float(np.mean(fold_eces)),
        "aggregated_probs": np.array(all_probs),
        "aggregated_labels": np.array(all_labels),
    }


def run_full_evaluation(model, X_test, y_test, model_name: str = "model") -> Dict:
    """Comprehensive evaluation: AUC, accuracy, calibration, bootstrap CIs."""
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "predict_uncertainty"):
        mean_preds, _ = model.predict_uncertainty(X_test)
        y_prob = mean_preds
    else:
        y_prob = model.predict(X_test)

    y_true = y_test.values if hasattr(y_test, "values") else y_test

    return {
        "model_name": model_name,
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, (y_prob > 0.5).astype(int))),
        "auc_ci": bootstrap_metric(y_true, y_prob, roc_auc_score, n_bootstrap=500),
        "calibration": compute_calibration_data(y_true, y_prob),
        "n_samples": len(y_true),
    }


def save_evaluation_results(results: Dict, output_dir: str = "models") -> str:
    """Save evaluation results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "eval_metrics.json")

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(filepath, "w") as f:
        json.dump(convert(results), f, indent=2)

    print(f"✅ Saved evaluation results to {filepath}")
    return filepath
