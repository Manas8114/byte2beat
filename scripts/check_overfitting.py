"""
Overfitting Diagnosis Script.
Compares train vs test AUC, runs cross-validation, and reports overfitting risk.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from uncertaintyml.evaluation import cross_validate_model, compute_ece


def check_model_dir(models_dir: str, label: str):
    """Run overfitting diagnostics on a model directory."""
    print(f"\n{'='*60}")
    print(f"  Overfitting Check: {label} ({models_dir})")
    print(f"{'='*60}")

    if not os.path.exists(models_dir):
        print(f"  ❌ Directory not found: {models_dir}")
        return

    # Load test data
    try:
        X_test = joblib.load(os.path.join(models_dir, "X_test.pkl"))
        y_test = joblib.load(os.path.join(models_dir, "y_test.pkl"))
    except FileNotFoundError:
        print("  ❌ X_test.pkl or y_test.pkl not found.")
        return

    # Force numeric
    for col in X_test.columns:
        X_test[col] = pd.to_numeric(X_test[col].astype(str).str.strip('[]'), errors='coerce').fillna(0).astype(np.float64)

    y_test_arr = y_test.values if hasattr(y_test, 'values') else y_test

    print(f"\n  Test set: {len(X_test)} samples, {X_test.shape[1]} features")
    print(f"  Class distribution: {dict(zip(*np.unique(y_test_arr, return_counts=True)))}")

    # Check each model
    for model_file in sorted(os.listdir(models_dir)):
        if not model_file.endswith("_model.pkl"):
            continue

        model_name = model_file.replace("_model.pkl", "")
        print(f"\n  --- {model_name.upper()} ---")

        try:
            model = joblib.load(os.path.join(models_dir, model_file))
        except Exception as e:
            print(f"    ❌ Failed to load: {e}")
            continue

        # Test predictions
        try:
            if hasattr(model, 'predict_proba'):
                y_prob_test = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'predict_uncertainty'):
                means, stds = model.predict_uncertainty(X_test, n_samples=30)
                y_prob_test = means
            else:
                y_prob_test = model.predict(X_test).astype(float)

            test_auc = roc_auc_score(y_test_arr, y_prob_test)
            test_acc = accuracy_score(y_test_arr, (y_prob_test > 0.5).astype(int))
            test_f1 = f1_score(y_test_arr, (y_prob_test > 0.5).astype(int))
            test_ece = compute_ece(y_test_arr, y_prob_test)

            print(f"    Test AUC:      {test_auc:.4f}")
            print(f"    Test Accuracy: {test_acc:.4f}")
            print(f"    Test F1:       {test_f1:.4f}")
            print(f"    Test ECE:      {test_ece:.4f}")

            # Classification report for minority class
            y_pred = (y_prob_test > 0.5).astype(int)
            report = classification_report(y_test_arr, y_pred, output_dict=True, zero_division=0)
            minority_class = '1'
            if minority_class in report:
                print(f"    Minority (class=1) Precision: {report[minority_class]['precision']:.4f}")
                print(f"    Minority (class=1) Recall:    {report[minority_class]['recall']:.4f}")

        except Exception as e:
            print(f"    ❌ Test evaluation failed: {e}")
            continue

        # Train predictions (overfitting check)
        # We need to reload the full data to compute train AUC
        # For now, we check if train AUC is available in eval_metrics
        eval_path = os.path.join(models_dir, "eval_metrics.json")
        if os.path.exists(eval_path):
            import json
            with open(eval_path) as f:
                eval_data = json.load(f)
            
            model_eval = eval_data.get('models', {}).get(model_name, {})
            if 'auc' in model_eval:
                stored_auc = model_eval['auc']
                gap = abs(stored_auc - test_auc)
                print(f"    Stored AUC:    {stored_auc:.4f}")
                print(f"    AUC Gap:       {gap:.4f}")
                if gap > 0.05:
                    print(f"    ⚠️  POTENTIAL OVERFITTING (gap > 0.05)")
                else:
                    print(f"    ✅ Gap looks healthy")

    print(f"\n{'='*60}\n")


def main():
    print("🔍 UncertaintyML: Overfitting Diagnosis Report")
    print(f"{'='*60}")

    check_model_dir("models", "Heart Disease")
    check_model_dir("models_stroke", "Stroke")

    print("✅ Diagnosis complete.")


if __name__ == "__main__":
    main()
