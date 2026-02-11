"""
UncertaintyPipeline — orchestrates the full train → evaluate → predict → explain flow.
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from uncertaintyml.models import UncertaintyModel, ModelRegistry
from uncertaintyml.data import DatasetAdapter, HeartDiseaseAdapter
from uncertaintyml.evaluation import run_full_evaluation, save_evaluation_results, compute_calibration_data
from uncertaintyml.interpret import ConceptBottleneck, NarrativeEngine
from uncertaintyml.cache import PredictionCache


@dataclass
class PipelineConfig:
    """Configuration for UncertaintyPipeline."""

    # Model settings
    models_to_train: List[str] = field(default_factory=lambda: ["xgboost", "tabpfn", "uncertainty"])
    uncertainty_epochs: int = 100
    uncertainty_lr: float = 0.005
    mc_samples: int = 30
    dropout_rate: float = 0.3
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])

    # Data settings
    test_size: float = 0.2
    random_state: int = 42
    tabpfn_max_samples: int = 1000

    # Output settings
    output_dir: str = "models"

    # Cache settings
    use_cache: bool = True
    cache_max_size: int = 10000
    cache_ttl_seconds: int = 3600
    cache_type: str = "memory"
    redis_url: Optional[str] = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    
    # Conformal settings
    calibration_size: float = 0.15  # Reserve 15% of training data for calibration
    conformal_alpha: float = 0.1    # 90% confidence target

    # Smart routing
    fast_path_threshold: float = 0.85


class UncertaintyPipeline:
    """
    End-to-end pipeline for uncertainty-aware medical risk assessment.

    Usage:
        config = PipelineConfig(models_to_train=["xgboost", "uncertainty"])
        pipe = UncertaintyPipeline(config)
        results = pipe.train(X, y, concept_map)
        prediction = pipe.predict(patient_data)
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.registry = ModelRegistry()
        self.trained_models: Dict[str, object] = {}
        self.conformal_models: Dict[str, Any] = {}  # Map: model_name -> ConformalCalibrator
        self.eval_results: Dict = {}
        self.concept_bottleneck: Optional[ConceptBottleneck] = None
        self.narrative_engine: Optional[NarrativeEngine] = None
        self.X_test = None
        self.y_test = None
        self._cache: Optional[PredictionCache] = None
        if self.config.use_cache:
            self._cache = PredictionCache(
                max_size=self.config.cache_max_size,
                ttl_seconds=self.config.cache_ttl_seconds,
                cache_type=self.config.cache_type,
                redis_url=self.config.redis_url
            )

    def train(self, X: pd.DataFrame, y: pd.Series, concept_map: Dict = None) -> Dict:
        """
        Train all configured models and evaluate them.

        Returns:
            Dict with evaluation results for each model.
        """
        from sklearn.model_selection import train_test_split
        from uncertaintyml.conformal import ConformalCalibrator

        # Setup interpretability
        if concept_map:
            self.concept_bottleneck = ConceptBottleneck(concept_map)
            self.narrative_engine = NarrativeEngine(self.concept_bottleneck)
        else:
            self.narrative_engine = NarrativeEngine()

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        
        # Split remaining training data for calibration if needed
        X_train_final, X_cal, y_train_final, y_cal = train_test_split(
            X_train, y_train, 
            test_size=self.config.calibration_size, 
            random_state=self.config.random_state, 
            stratify=y_train
        )
        # Re-assign for evaluation consistency
        X_train = X_train_final
        y_train = y_train_final
        self.X_test = X_test
        self.y_test = y_test

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": X.shape[1],
            "models": {},
        }

        for model_name in self.config.models_to_train:
            print(f"\n🔧 Training {model_name}...")
            try:
                if model_name == "uncertainty":
                    model = self.registry.create(
                        model_name,
                        epochs=self.config.uncertainty_epochs,
                        lr=self.config.uncertainty_lr,
                        hidden_dims=self.config.hidden_dims,
                        dropout_rate=self.config.dropout_rate,
                        n_inference_samples=self.config.mc_samples,
                    )
                elif model_name == "tabpfn":
                    model = self.registry.create(model_name)
                    if len(X_train) > self.config.tabpfn_max_samples:
                        X_train_sub = X_train.iloc[: self.config.tabpfn_max_samples]
                        y_train_sub = y_train.iloc[: self.config.tabpfn_max_samples]
                    else:
                        X_train_sub, y_train_sub = X_train, y_train
                    model.fit(X_train_sub, y_train_sub)
                    self.trained_models[model_name] = model
                    
                    # Calibrate
                    cal = ConformalCalibrator(model, method="score", cv="prefit")
                    cal.fit(X_cal, y_cal)
                    self.conformal_models[model_name] = cal
                    eval_result = run_full_evaluation(model, X_test, y_test, model_name=model_name)
                    all_results["models"][model_name] = eval_result
                    print(f"  ✅ AUC: {eval_result['auc']:.4f}")
                    continue
                else:
                    model = self.registry.create(model_name)

                model.fit(X_train, y_train)
                self.trained_models[model_name] = model

                # Calibrate
                cal = ConformalCalibrator(model, method="score", cv="prefit")
                cal.fit(X_cal, y_cal)
                self.conformal_models[model_name] = cal

                # Evaluate
                if model_name == "uncertainty":
                    mean_preds, std_preds = model.predict_uncertainty(X_test, n_samples=self.config.mc_samples)
                    y_test_arr = y_test.values if hasattr(y_test, "values") else y_test
                    errors = np.abs(mean_preds - y_test_arr)
                    unc_error_corr = float(np.corrcoef(std_preds, errors)[0, 1])
                    cal = compute_calibration_data(y_test_arr, mean_preds)
                    all_results["models"][model_name] = {
                        "model_name": "MC Dropout MLP",
                        "mean_uncertainty": float(np.mean(std_preds)),
                        "max_uncertainty": float(np.max(std_preds)),
                        "uncertainty_error_correlation": unc_error_corr,
                        "calibration": cal,
                    }
                    print(f"  ✅ Mean σ: {np.mean(std_preds):.4f}")
                else:
                    eval_result = run_full_evaluation(model, X_test, y_test, model_name=model_name)
                    all_results["models"][model_name] = eval_result
                    print(f"  ✅ AUC: {eval_result['auc']:.4f}")

            except Exception as e:
                print(f"  ❌ {model_name} failed: {e}")

        self.eval_results = all_results

        # Invalidate cache after retraining
        if self._cache is not None:
            self._cache.invalidate()

        return all_results

    def _align_features(self, model, X: pd.DataFrame) -> pd.DataFrame:
        """Align features to model's expected order if available."""
        target_features = getattr(model, "feature_names_in_", None)
        
        # If model doesn't have features, try to find a reference model that does
        if target_features is None:
            for m in self.trained_models.values():
                if hasattr(m, "feature_names_in_"):
                    target_features = m.feature_names_in_
                    break
        
        if target_features is not None:
            # Check if all features exist
            missing = set(target_features) - set(X.columns)
            if not missing:
                 return X[target_features]
        
        return X

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Run full evaluation suite."""
        self.X_test = X_test
        self.y_test = y_test

        results = {}
        for name, model in self.trained_models.items():
            try:
                X_aligned = self._align_features(model, X_test)
                results[name] = run_full_evaluation(model, X_aligned, y_test, model_name=name)
            except Exception as e:
                print(f"Error evaluating {name}: {e}")

        # Add Conformal Metrics
        results["conformal"] = {}
        for name, cal in self.conformal_models.items():
            X_aligned = self._align_features(cal.estimator, X_test)
            results["conformal"][name] = cal.evaluate(X_aligned, y_test, alpha=self.config.conformal_alpha)

        self.eval_results = results
        return results

    def predict(self, patient_data: pd.DataFrame, model_name: str = "uncertainty") -> Dict:
        """
        Predict risk for a patient with uncertainty and narrative explanation.

        Performance optimizations:
            - LRU cache lookup (sub-ms for repeated queries)
            - Smart routing: fast path (XGBoost) for high-confidence cases

        Returns:
            Dict with risk_score, uncertainty, confidence_level, narrative,
            model_used, cache_hit, route_taken
        """
        model = self.trained_models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not trained. Available: {list(self.trained_models.keys())}")

        if isinstance(patient_data, pd.Series):
            patient_data = patient_data.to_frame().T
        
        # Align features
        patient_data = self._align_features(model, patient_data)

        # --- Cache lookup ---
        cache_key = None
        if self._cache is not None:
            features = patient_data.iloc[0].to_dict()
            cache_key = self._cache.make_key({**features, "_model": model_name})
            cached = self._cache.get(cache_key)
            if cached is not None:
                cached["cache_hit"] = True
                return cached

        # --- Smart routing: fast path ---
        route_taken = "direct"
        xgb_model = self.trained_models.get("xgboost")
        if (
            model_name in ("uncertainty", "unc")
            and xgb_model is not None
            and hasattr(xgb_model, "predict_proba")
        ):
            xgb_proba = xgb_model.predict_proba(patient_data)
            xgb_confidence = float(max(xgb_proba[0]))

            if xgb_confidence >= self.config.fast_path_threshold:
                risk = float(xgb_proba[0, 1]) * 100
                uncertainty = (1 - xgb_confidence) * 100
                route_taken = "fast_path"
            else:
                mean_pred, std_pred = model.predict_uncertainty(patient_data)
                risk = float(mean_pred[0]) * 100
                uncertainty = float(std_pred[0]) * 100
                route_taken = "slow_path"
        elif hasattr(model, "predict_uncertainty"):
            mean_pred, std_pred = model.predict_uncertainty(patient_data)
            risk = float(mean_pred[0]) * 100
            uncertainty = float(std_pred[0]) * 100
        elif hasattr(model, "predict_proba"):
            proba = model.predict_proba(patient_data)
            risk = float(proba[0, 1]) * 100
            uncertainty = 0.0
        else:
            pred = model.predict(patient_data)
            risk = float(pred[0]) * 100
            uncertainty = 0.0

        confidence = "High" if uncertainty < 10 else ("Medium" if uncertainty < 20 else "Low")

        narrative = ""
        if self.narrative_engine:
            narrative = self.narrative_engine.generate(
                patient_data.iloc[0] if isinstance(patient_data, pd.DataFrame) else patient_data,
                risk, uncertainty,
            )

        result = {
            "risk_score": round(risk, 2),
            "uncertainty": round(uncertainty, 2),
            "confidence_level": confidence,
            "narrative": narrative,
            "model_used": model_name,
            "cache_hit": False,
            "route_taken": route_taken,
        }

        # 3. Conformal Prediction Set
        if model_name in self.conformal_models:
             _, y_sets = self.conformal_models[model_name].predict(patient_data, alpha=self.config.conformal_alpha)
             # y_sets is boolean array (n_samples, n_classes). True -> in set.
             # We want list of indices.
             if y_sets.ndim == 2:
                 in_set = [i for i, is_in in enumerate(y_sets[0]) if is_in]
                 result["conformal_set"] = in_set
                 result["conformal_alpha"] = self.config.conformal_alpha

        # --- Cache store ---
        if self._cache is not None and cache_key is not None:
            self._cache.put(cache_key, result)

        return result

    def save(self, output_dir: str = None):
        """Save all trained models and evaluation results."""
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        for name, model in self.trained_models.items():
            path = os.path.join(output_dir, f"{name}_model.pkl")
            joblib.dump(model, path)
            print(f"  ✅ Saved: {path}")

        for name, cal in self.conformal_models.items():
            joblib.dump(cal, os.path.join(output_dir, f"{name}_conformal.pkl"))

        if self.X_test is not None:
            joblib.dump(self.X_test, os.path.join(output_dir, "X_test.pkl"))
            joblib.dump(self.y_test, os.path.join(output_dir, "y_test.pkl"))

        if self.eval_results:
            save_evaluation_results(self.eval_results, output_dir)

        model_info = {
            "version": "2.0.0",
            "trained_at": datetime.now().isoformat(),
            "models_trained": list(self.trained_models.keys()),
        }
        with open(os.path.join(output_dir, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)

        if self.concept_bottleneck:
            with open(os.path.join(output_dir, "concept_map.json"), "w") as f:
                json.dump(self.concept_bottleneck.concept_map, f, indent=2)

    @classmethod
    def load(cls, models_dir: str = "models") -> "UncertaintyPipeline":
        """Load a previously saved pipeline."""
        pipe = cls()

        import warnings
        for f in os.listdir(models_dir):
            if f.endswith("_model.pkl"):
                name = f.replace("_model.pkl", "")
                try:
                    pipe.trained_models[name] = joblib.load(os.path.join(models_dir, f))
                except Exception as e:
                    warnings.warn(f"Could not load model '{name}' from {f}: {e}")
            
            if f.endswith("_conformal.pkl"):
                name = f.replace("_conformal.pkl", "")
                try:
                    pipe.conformal_models[name] = joblib.load(os.path.join(models_dir, f))
                except Exception as e:
                    warnings.warn(f"Could not load conformal '{name}': {e}")

        x_path = os.path.join(models_dir, "X_test.pkl")
        y_path = os.path.join(models_dir, "y_test.pkl")
        if os.path.exists(x_path):
            pipe.X_test = joblib.load(x_path)
        if os.path.exists(y_path):
            pipe.y_test = joblib.load(y_path)

        eval_path = os.path.join(models_dir, "eval_metrics.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                pipe.eval_results = json.load(f)

        # Restore interpretability from saved concept_map or default
        concept_path = os.path.join(models_dir, "concept_map.json")
        if os.path.exists(concept_path):
            with open(concept_path) as f:
                concept_map = json.load(f)
            pipe.concept_bottleneck = ConceptBottleneck(concept_map)
            pipe.narrative_engine = NarrativeEngine(pipe.concept_bottleneck)
        else:
            pipe.narrative_engine = NarrativeEngine()

        return pipe
