import sys
import os
import pandas as pd
import numpy as np
import torch
import joblib

# Add project root and src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath('src'))

try:
    from uncertaintyml.models import UncertaintyModel
except ImportError:
    from utils_model import UncertaintyModel

def test_trust_mechanics():
    print("🧪 Starting Trust Test (Safety Check)...")
    
    # Load Model
    model_path = 'models/uncertainty_model.pkl'
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}. Run 'run_experiment.py' first.")
        return False

    try:
        unc_model = joblib.load(model_path)
        print("✅ Uncertainty Model Loaded.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Create Garbage Data (Random Noise)
    try:
        X_test = joblib.load('models/X_test.pkl')
        
        # Generate Random Noise (Uniform distribution) in plausible-ish but random ranges
        # This prevents "saturation" that might happen with simple fixed large values.
        n_garbage = 20
        garbage_df = pd.DataFrame(index=range(n_garbage), columns=X_test.columns)
        
        for col in X_test.columns:
            # Random uniform noise. 
            # We want to cover the input space and go slightly beyond.
            # Assuming scaler is fit on data, extreme values might get scaled to +/- 10.
            # We want values that are somewhat uniform to trigger variance.
            garbage_df[col] = np.random.uniform(-500, 500, n_garbage)
            
        input_data = garbage_df
        print(f"✅ Generated {n_garbage} rows of random noise data.")
        
    except Exception as e:
        print(f"❌ Failed to generate noise data: {e}")
        return False

    # Predict
    print(f"\n🧐 Injecting Noise...")
    # print(input_data.head())
    
    mean_preds, std_preds = unc_model.predict_uncertainty(input_data, n_samples=50)
    
    avg_risk = np.mean(mean_preds) * 100
    avg_uncertainty = np.mean(std_preds) * 100
    
    print(f"\n📊 Diagnostics (Average across {n_garbage} samples):")
    print(f"   Risk Score:       {avg_risk:.2f}% (Should be anything)")
    print(f"   Uncertainty (σ):  {avg_uncertainty:.2f}% (Target: >10%)")
    
    # Assertions
    # We expect high uncertainty. 
    # With random noise, the ensemble members should disagree significantly.
    
    if avg_uncertainty > 10.0:
        print(f"\n✅ PASS: High Uncertainty Detected ({avg_uncertainty:.2f}% > 10%).")
        print("   Safety Mechanism: ACTIVE.")
        return True
    else:
        print(f"\n❌ FAIL: Uncertainty too low ({avg_uncertainty:.2f}%).")
        print("   The model is overconfident even on random noise. Safety Mechanism: INACTIVE.")
        return False

if __name__ == "__main__":
    success = test_trust_mechanics()
    sys.exit(0 if success else 1)
