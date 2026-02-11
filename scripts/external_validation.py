"""
External Validation Script for UncertaintyML.
Tests model robustness on Out-of-Distribution (OOD) data.
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from uncertaintyml.pipeline import UncertaintyPipeline
from uncertaintyml.data import HeartDiseaseAdapter

# Ensure we can import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_ood_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesize OOD data by shifting demographics and adding noise.
    Simulates a 'Geriatric' or 'Different Hospital' cohort.
    """
    print("Generating OOD Data (Synthetic Shift)...")
    df_ood = df.copy()
    
    # Shift 1: Older population (Age + 15 years, capped at 90)
    if 'Age' in df_ood.columns:
        df_ood['Age'] = df_ood['Age'] + 15
        df_ood['Age'] = df_ood['Age'].clip(upper=90)
        
    # Shift 2: Higher baseline risk (BP + 10, Cholesterol + 20)
    if 'RestingBP' in df_ood.columns:
        df_ood['RestingBP'] = df_ood['RestingBP'] + np.random.normal(10, 5, len(df_ood))
    if 'Cholesterol' in df_ood.columns:
        df_ood['Cholesterol'] = df_ood['Cholesterol'] + np.random.normal(20, 10, len(df_ood))
        
    # Shift 3: Noise in continuous features
    for col in ['MaxHR', 'Oldpeak']:
        if col in df_ood.columns:
            noise = np.random.normal(0, df_ood[col].std() * 0.2, len(df_ood))
            df_ood[col] = df_ood[col] + noise
            
    return df_ood

def main():
    # 1. Load Data
    print("1. Loading Original Data...")
    adapter = HeartDiseaseAdapter()
    processed_path = 'Data/Heart Attack/heart_processed.csv'
    base_path = 'Data/Cardiac Failure/cardio_base.csv'
    
    if not os.path.exists(processed_path):
        print(f"❌ Data not found at {processed_path}")
        return

    # Using the updated load signature: primary, *extra
    if os.path.exists(base_path):
        X, y, _ = adapter.load(processed_path, base_path)
    else:
        X, y, _ = adapter.load(processed_path)
        
    # 2. Generate OOD Data
    X_ood = generate_ood_data(X)
    y_ood = y.copy() # Labels remain same for now (concept shift vs covariate shift)
    
    # 3. Load Pipeline
    print("\n2. Loading Model Pipeline...")
    try:
        pipe = UncertaintyPipeline.load('models')
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        return

    # 4. Evaluate
    print("\n3. Running Evaluation on OOD Data...")
    
    # We want to see how performance degrades
    results = pipe.evaluate(X_ood, y_ood)
    
    print("\n" + "="*40)
    print("EXTERNAL VALIDATION RESULTS (OOD)")
    print("="*40)
    
    for model_name, metrics in results.items():
        if model_name == "conformal":
            print(f"\n[Conformal Prediction]")
            for m_name, c_metrics in metrics.items():
                 print(f"  {m_name}:")
                 print(f"    Coverage: {c_metrics.get('coverage', 'N/A')}")
                 print(f"    Avg Set Size: {c_metrics.get('avg_set_size', 'N/A')}")
            continue
            
        if model_name in ["timestamp", "n_train", "n_test", "n_features", "models"]:
             continue
             
        # Handle the structure of results: it returns 'all_results' which has 'models' key
        # But evaluate() sets pipe.eval_results = all_results
        pass

    # The result structure from pipeline.evaluate returns the full dict.
    # Let's parse 'models' key
    if "models" in results:
        for m_name, m_res in results["models"].items():
            print(f"\nModel: {m_name}")
            if "auc" in m_res:
                print(f"  AUC: {m_res['auc']:.4f}")
            if "accuracy" in m_res:
                print(f"  Accuracy: {m_res['accuracy']:.4f}")
            if "calibration" in m_res:
                print(f"  ECE: {m_res['calibration']['ece']:.4f}")

if __name__ == "__main__":
    main()
