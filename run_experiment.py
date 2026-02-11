"""
End-to-End Experiment Runner for UncertaintyML.
Now uses the production-grade UncertaintyPipeline for full integration testing.
"""

import sys
import os
import pandas as pd
import json
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uncertaintyml.pipeline import PipelineConfig, UncertaintyPipeline
from uncertaintyml.data import load_and_preprocess_data

def main():
    print("=" * 60)
    print("🚀 UncertaintyML: End-to-End Experiment Pipeline")
    print("=" * 60)

    # 1. Load Data
    print("\n[1/4] Loading Data...")
    try:
        X, y, concept_map = load_and_preprocess_data(
            'Data/Heart Attack/heart_processed.csv', 
            base_path='Data/Cardiac Failure/cardio_base.csv'
        )
        print(f"✅ Loaded {len(X)} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"❌ Data load failed: {e}")
        return

    # 2. Configure Pipeline
    print("\n[2/4] Configuring Pipeline...")
    config = PipelineConfig(
        models_to_train=["xgboost", "uncertainty"], # TabPFN optional/slow
        uncertainty_epochs=20,      # Fast run
        uncertainty_lr=0.005,
        mc_samples=10,
        dropout_rate=0.3,
        test_size=0.2,
        calibration_size=0.15,      # For Conformal Prediction
        output_dir="models",
        use_cache=True,
        cache_type="memory"
    )
    
    pipeline = UncertaintyPipeline(config)

    # 3. Train & Evaluate
    print("\n[3/4] Training Pipeline (including Conformal Calibration)...")
    try:
        # train() handles splitting, training, calibrating, and initial eval
        results = pipeline.train(X, y, concept_map)
        print("✅ Training Complete.")
        print(f"Models trained: {results['models']}")
        
        # Run detailed evaluation on test set
        print("\nRunning Evaluation...")
        eval_results = pipeline.evaluate(pipeline.X_test, pipeline.y_test)
        
        # specific check for Conformal
        if "conformal" in eval_results:
             print("\nConformal Prediction Metrics:")
             for m, metrics in eval_results["conformal"].items():
                 print(f"  {m}: Coverage={metrics.get('coverage')}, Width={metrics.get('avg_set_size')}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Save Artifacts
    print("\n[4/4] Saving Pipeline...")
    try:
        pipeline.save()
        print(f"✅ Pipeline saved to {config.output_dir}")
    except Exception as e:
         print(f"❌ Save failed: {e}")

    print("\n🎉 Experiment Success!")

if __name__ == "__main__":
    main()
