import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Add src to path
sys.path.append(os.path.abspath('src'))
from utils_data import load_and_preprocess_data
from utils_model import get_xgboost, UncertaintyModel

def retrain_light():
    print("🚀 Retraining Lightweight Models (XGB + UNC)...")
    
    # 1. Load Data
    DATA_PATH_PROC = 'Data/Heart Attack/heart_processed.csv'
    DATA_PATH_BASE = 'Data/Cardiac Failure/cardio_base.csv'
    
    X, y, _ = load_and_preprocess_data(DATA_PATH_PROC, base_path=DATA_PATH_BASE)
    print(f"Data Loaded. Samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    # 2. XGBoost
    print("Training XGBoost...")
    try:
        xgb = get_xgboost()
        xgb.fit(X_train, y_train)
        print("XGB Score:", xgb.score(X_test, y_test))
        joblib.dump(xgb, f'{models_dir}/xgb_model.pkl')
    except Exception as e:
        print(f"XGB Failed: {e}")

    # 3. Uncertainty Model
    print("Training Uncertainty Model (MCDropout + StandardScaler)...")
    try:
        unc_model = UncertaintyModel(epochs=50, lr=0.01) 
        unc_model.fit(X_train, y_train)
        joblib.dump(unc_model, f'{models_dir}/uncertainty_model.pkl')
    except Exception as e:
         print(f"UNC Failed: {e}")

    # 4. Save Test Data (Critical for scaling checks)
    joblib.dump(X_test, f'{models_dir}/X_test.pkl')
    joblib.dump(y_test, f'{models_dir}/y_test.pkl')
    print("✅ All lightweight models saved.")

if __name__ == "__main__":
    retrain_light()
