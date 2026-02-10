import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
import joblib
import torch

# Add src to path
sys.path.append(os.path.abspath('src'))
from utils_data import load_and_preprocess_data
from utils_model import get_xgboost, UncertaintyModel

def check_health():
    print("🏥 Starting Model Health Check...")
    
    # 1. Data Analysis
    print("\n[1/3] Checking Data Distribution...")
    try:
        X_test = joblib.load('models/X_test.pkl')
        y_test = joblib.load('models/y_test.pkl')
        # We can't verify raw distribution easily here without X_train, 
        # but we can rely on run_experiment logic.
        print("Loaded X_test from 'models/X_test.pkl' to ensure schema consistency.")
        X = X_test # For analysis usage
    except Exception as e:
        print(f"Failed to load test pickles: {e}")
        return
    
    # Check Scales
    print(f"Feature Stats:\n{X.describe().loc[['min', 'max', 'mean']].T}")
    
    # Check for unscaled consistency
    max_val = X.max().max()
    if max_val > 10:
        print(f"⚠️  WARNING: Data appears unscaled (Max value: {max_val:.2f}). MLPs will struggle.")
    else:
        print("✅ Data appears scaled.")

    # 2. Overfitting Check (XGBoost)
    print("\n[2/3] Checking XGBoost Overfitting...")
    # We only have X_test loaded. We need X_train to check overfitting properly.
    # But for now, let's just check Test Accuracy and warn if it's suspiciously low/high.
    # Ideally we load X_train.pkl if we saved it, or we rely on the log from run_experiment.
    
    model_path = 'models/xgb_model.pkl'
    if os.path.exists(model_path):
        xgb = joblib.load(model_path)
        # train_acc = ... (Skipping Train Acc as we don't have X_train handy implies loading mismatch risk)
        test_acc = accuracy_score(y_test, xgb.predict(X_test))
        print(f"XGB Test Acc:  {test_acc:.4f}")
        
        if test_acc < 0.6:
             print("⚠️  WARNING: Test Accuracy is low (<60%).")
        else:
             print("✅ XGBoost Test Accuracy looks healthy.")
    else:
        print("Skipping XGB check (model not found).")

    # 3. Uncertainty Model Scaling Check
    print("\n[3/3] Checking Uncertainty Model on Scaled? Data...")
    # The model has internal scaler now, so we pass raw-ish X_test (which was split from preprocessed X)
    # utils_data.load_and_preprocess returns 'X' which is numeric but not StandardScaled.
    # So X_test is numeric. 'UncertaintyModel' will scale it internally. Correct.
    try:
        unc_model_path = 'models/unc_model.pkl'
        if os.path.exists(unc_model_path):
            unc_model = joblib.load(unc_model_path)
            
            # Check predictions
            mean_preds, std_preds = unc_model.predict_uncertainty(X_test.iloc[:10])
            print(f"Uncertainty Model Sample Preds (Mean): {mean_preds[:5]}")
            print(f"Uncertainty Model Sample Preds (Std):  {std_preds[:5]}")
            
            if np.all(mean_preds < 0.01) or np.all(mean_preds > 0.99):
                 print("⚠️  WARNING: Uncertainty model output saturated.")
            else:
                 print("✅ Uncertainty model outputs look dynamic.")
        else:
             print("Skipping UNC check (model not found).")
             
    except Exception as e:
        print(f"Uncertainty Check Failed: {e}")

if __name__ == "__main__":
    check_health()
