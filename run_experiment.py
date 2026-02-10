
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath('src'))

from utils_data import load_and_preprocess_data, get_concept_map
from utils_model import get_xgboost, get_tabpfn, UncertaintyModel
from evaluation import run_full_evaluation, save_evaluation_results, compute_calibration_data

try:
    from huggingface_hub import login
    # Set Hugging Face Token for TabPFN (Use env var for security)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: HF_TOKEN not set. TabPFN might fail to load.")
except ImportError:
    print("huggingface_hub not installed. TabPFN may not work.")

def main():
    print("=" * 60)
    print("🫀 Uncertainty-Aware Cardiac Risk Assessment")
    print("   Rigorous Evaluation Pipeline")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1/6] Loading and Merging Datasets...")
    DATA_PATH_PROC = 'Data/Heart Attack/heart_processed.csv'
    DATA_PATH_BASE = 'Data/Cardiac Failure/cardio_base.csv'
    
    if not os.path.exists(DATA_PATH_PROC):
        print(f"Error: {DATA_PATH_PROC} not found.")
        return
        
    try:
        X, y, concept_map = load_and_preprocess_data(DATA_PATH_PROC, base_path=DATA_PATH_BASE)
        print(f"✅ Data Loaded. Total Samples: {len(X)}, Features: {X.shape[1]}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Storage for evaluation results
    all_eval_results = {
        'timestamp': datetime.now().isoformat(),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X.shape[1],
        'models': {}
    }
    
    # 2. Baseline Model (XGBoost)
    print("\n[2/6] Training Baseline (XGBoost)...")
    xgb = None
    try:
        xgb = get_xgboost()
        xgb.fit(X_train, y_train)
        print(">>> XGBoost Performance:")
        print(classification_report(y_test, xgb.predict(X_test)))
        
        # Rigorous Evaluation
        xgb_eval = run_full_evaluation(xgb, X_test, y_test, model_name="XGBoost")
        all_eval_results['models']['xgboost'] = xgb_eval
        print(f"📊 AUC: {xgb_eval['auc']:.4f} (95% CI: {xgb_eval['auc_ci']['lower']:.4f}-{xgb_eval['auc_ci']['upper']:.4f})")
        print(f"📊 ECE: {xgb_eval['calibration']['ece']:.4f} (Lower is better)")
        
    except Exception as e:
        print(f"XGBoost Failed: {e}")

    # 3. Novel Model (TabPFN)
    print("\n[3/6] Training Novel Model (TabPFN)...")
    tabpfn = None
    try:
        if len(X_train) > 1000:
            print(f"Dataset large ({len(X_train)}), subsampling 1000 for TabPFN...")
            X_train_sub = X_train.iloc[:1000]
            y_train_sub = y_train.iloc[:1000]
        else:
            X_train_sub, y_train_sub = X_train, y_train
            
        tabpfn = get_tabpfn()
        tabpfn.fit(X_train_sub, y_train_sub)
        print(">>> TabPFN Performance:")
        print(classification_report(y_test, tabpfn.predict(X_test)))
        
        # Rigorous Evaluation
        tabpfn_eval = run_full_evaluation(tabpfn, X_test, y_test, model_name="TabPFN")
        all_eval_results['models']['tabpfn'] = tabpfn_eval
        print(f"📊 AUC: {tabpfn_eval['auc']:.4f} (95% CI: {tabpfn_eval['auc_ci']['lower']:.4f}-{tabpfn_eval['auc_ci']['upper']:.4f})")
        print(f"📊 ECE: {tabpfn_eval['calibration']['ece']:.4f}")
        
    except Exception as e:
        print(f"TabPFN Failed: {e}")
        print("Note: TabPFN requires 'tabpfn' package.")

    # 4. Uncertainty Model
    print("\n[4/6] Training Uncertainty Model (MC Dropout)...")
    unc_model = None
    try:
        unc_model = UncertaintyModel(epochs=100, lr=0.005)
        unc_model.fit(X_train, y_train)
        
        mean_preds, std_preds = unc_model.predict_uncertainty(X_test, n_samples=50)
        print(f"✅ Uncertainty Estimation Complete.")
        print(f"📊 Mean Uncertainty (σ): {np.mean(std_preds):.4f}")
        print(f"📊 Max Uncertainty (σ):  {np.max(std_preds):.4f}")
        
        # Calibration for uncertainty model
        y_test_arr = y_test.values if hasattr(y_test, 'values') else y_test
        unc_calibration = compute_calibration_data(y_test_arr, mean_preds)
        
        # Uncertainty Validation: Does high uncertainty correlate with errors?
        errors = np.abs(mean_preds - y_test_arr)
        unc_error_corr = float(np.corrcoef(std_preds, errors)[0, 1])
        print(f"📊 Uncertainty-Error Correlation: {unc_error_corr:.4f} (>0 means uncertainty is meaningful)")
        
        all_eval_results['models']['uncertainty'] = {
            'model_name': 'MC Dropout MLP',
            'mean_uncertainty': float(np.mean(std_preds)),
            'max_uncertainty': float(np.max(std_preds)),
            'uncertainty_error_correlation': unc_error_corr,
            'calibration': unc_calibration
        }
        
    except Exception as e:
        print(f"Uncertainty Model Failed: {e}")

    # 5. Save Models
    print("\n[5/6] Saving Models...")
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        if xgb:
            joblib.dump(xgb, f'{models_dir}/xgb_model.pkl')
            print(f"  ✅ Saved: xgb_model.pkl")
            
        if tabpfn:
            joblib.dump(tabpfn, f'{models_dir}/tabpfn_model.pkl')
            print(f"  ✅ Saved: tabpfn_model.pkl")
            
        if unc_model:
            joblib.dump(unc_model, f'{models_dir}/unc_model.pkl')
            print(f"  ✅ Saved: unc_model.pkl")
            
        joblib.dump(X_test, f'{models_dir}/X_test.pkl')
        joblib.dump(y_test, f'{models_dir}/y_test.pkl')
        print(f"  ✅ Saved: X_test.pkl, y_test.pkl")
        
    except Exception as e:
        print(f"Error saving models: {e}")

    # 6. Save Evaluation Results (for Dashboard)
    print("\n[6/6] Saving Evaluation Metrics...")
    save_evaluation_results(all_eval_results, models_dir)
    
    # Save Model Info (Versioning)
    model_info = {
        'version': '2.0.0',
        'trained_at': datetime.now().isoformat(),
        'n_samples': len(X),
        'n_features': X.shape[1],
        'best_auc': max([m.get('auc', 0) for m in all_eval_results['models'].values()]),
        'models_trained': list(all_eval_results['models'].keys())
    }
    with open(f'{models_dir}/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"  ✅ Saved: model_info.json")
    
    print("\n" + "=" * 60)
    print("🎉 Experiment Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
