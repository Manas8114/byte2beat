"""
End-to-End Experiment Runner for UncertaintyML applied to Stroke Dataset.
Uses the production-grade UncertaintyPipeline.
"""

import sys
import os
import pandas as pd
import json
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uncertaintyml.pipeline import PipelineConfig, UncertaintyPipeline
from uncertaintyml.adapters.stroke import StrokeAdapter
from uncertaintyml.evaluation import cross_validate_model

def main():
    print("=" * 60)
    print("🚀 UncertaintyML: End-to-End Stroke Experiment Pipeline")
    print("=" * 60)

    # 1. Load Data
    print("\n[1/4] Loading Data...")
    try:
        adapter = StrokeAdapter()
        X, y, concept_map = adapter.load('Data/Stroke/stroke_dataset.csv')
        print(f"✅ Loaded {len(X)} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"❌ Data load failed: {e}")
        return

    # 2. Configure Pipeline
    print("\n[2/4] Configuring Pipeline...")
    config = PipelineConfig(
        models_to_train=["xgboost", "uncertainty"], 
        uncertainty_epochs=20,      
        uncertainty_lr=0.005,
        mc_samples=10,
        dropout_rate=0.3,
        test_size=0.2,
        calibration_size=0.15,      
        output_dir="models_stroke", # Save to new dir
        use_cache=True,
        cache_type="memory"
    )
    
    pipeline = UncertaintyPipeline(config)

    # 3. Train & Evaluate
    print("\n[3/4] Training Pipeline...")
    try:
        results = pipeline.train(X, y, concept_map)
        print("✅ Training Complete.")
        print(f"Models trained: {results['models']}")
        
        # Run detailed evaluation on test set
        print("\nRunning Evaluation...")
        eval_results = pipeline.evaluate(pipeline.X_test, pipeline.y_test)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Save Artifacts
    print("\n[4/5] Saving Pipeline...")
    try:
        pipeline.save()
        print(f"✅ Pipeline saved to {config.output_dir}")
    except Exception as e:
         print(f"❌ Save failed: {e}")

    # 5. Cross-Validation
    print("\n[5/5] Cross-Validation & Overfitting Check...")
    try:
        import xgboost as xgb
        cv_results = cross_validate_model(
            xgb.XGBClassifier,
            X, y, n_splits=5,
            eval_metric='logloss', random_state=42,
            max_depth=4, n_estimators=200, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=3
        )
        print(f"  CV AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
        print(f"  CV Acc: {cv_results['mean_acc']:.4f} ± {cv_results['std_acc']:.4f}")
        if cv_results['std_auc'] > 0.05:
            print("  [!] HIGH VARIANCE across folds")
        else:
            print("  [OK] Fold variance is healthy")
    except Exception as e:
        print(f"  [!] CV check skipped: {e}")

    print("\nExperiment Success!")

if __name__ == "__main__":
    main()
