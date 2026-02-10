import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# Add src to path
sys.path.append(os.path.abspath('src'))
from utils_data import load_and_preprocess_data
from utils_model import UncertaintyModel

def retrain_uncertainty():
    print("🚀 Retraining Uncertainty Model ONLY...")
    
    # 1. Load Data
    DATA_PATH_PROC = 'Data/Heart Attack/heart_processed.csv'
    DATA_PATH_BASE = 'Data/Cardiac Failure/cardio_base.csv'
    
    try:
        X, y, _ = load_and_preprocess_data(DATA_PATH_PROC, base_path=DATA_PATH_BASE)
        print(f"Data Loaded. Samples: {len(X)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Retrain Uncertainty Model (New Logic with Scaler)
    print("Training Uncertainty Model (MCDropout + StandardScaler)...")
    try:
        # Increase epochs slightly to ensure convergence with scaler
        unc_model = UncertaintyModel(epochs=50, lr=0.01) 
        unc_model.fit(X_train, y_train)
        
        # Quick validation
        mean, std = unc_model.predict_uncertainty(X_test.iloc[:10])
        print(f"Sample Uncertainties: {std}")
        
        # Save
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(unc_model, f'{models_dir}/unc_model.pkl')
        print(f"✅ Saved model to {models_dir}/unc_model.pkl")
        
    except Exception as e:
        print(f"❌ Training Failed: {e}")

if __name__ == "__main__":
    retrain_uncertainty()
