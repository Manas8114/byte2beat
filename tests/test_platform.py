"""
Platform verification tests for the uncertaintyml package.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_package_imports():
    from uncertaintyml import (
        UncertaintyPipeline, UncertaintyModel, ModelRegistry,
        ConceptBottleneck, NarrativeEngine, DatasetAdapter, HeartDiseaseAdapter,
    )
    assert UncertaintyPipeline is not None
    assert UncertaintyModel is not None


def test_model_registry():
    from uncertaintyml.models import ModelRegistry
    reg = ModelRegistry()
    assert "uncertainty" in reg.available_models
    assert "xgboost" in reg.available_models

    model = reg.create("uncertainty", epochs=1)
    assert model is not None


def test_concept_bottleneck():
    from uncertaintyml.interpret import ConceptBottleneck
    cb = ConceptBottleneck({
        "Vitals": ["BP", "HR"],
        "Demographics": ["Age"],
    })
    scores = cb.aggregate({"BP": 0.5, "HR": 0.3, "Age": 0.2})
    assert abs(sum(scores.values()) - 1.0) < 0.01
    assert cb.get_dominant_concept({"BP": 0.5, "HR": 0.3, "Age": 0.2}) == "Vitals"


def test_narrative_engine():
    import pandas as pd
    from uncertaintyml.interpret import NarrativeEngine
    engine = NarrativeEngine()
    narrative = engine.generate(
        pd.Series({"Age": 55, "Cholesterol": 250, "RestingBP": 140, "MaxHR": 120, "FastingBS": 1}),
        risk_score=72.0,
        uncertainty_score=8.0,
    )
    assert "risk" in narrative.lower()
    assert len(narrative) > 50


def test_pipeline_load_and_predict():
    from uncertaintyml.pipeline import UncertaintyPipeline
    if not os.path.exists("models/uncertainty_model.pkl"):
        import pytest
        pytest.skip("Pre-trained models not found")

    pipe = UncertaintyPipeline.load("models")
    assert len(pipe.trained_models) > 0

    result = pipe.predict(pipe.X_test.iloc[:1], model_name="uncertainty")
    assert "risk_score" in result
    assert "uncertainty" in result
    assert "narrative" in result
    assert 0 <= result["risk_score"] <= 100


def test_heart_disease_adapter():
    from uncertaintyml.data import HeartDiseaseAdapter
    adapter = HeartDiseaseAdapter()
    cmap = adapter.get_concept_map()
    assert "Demographics" in cmap
    assert "Vitals" in cmap
    assert "Clinical" in cmap


def test_api_schemas():
    from api.schemas import PatientInput, PredictionResponse
    patient = PatientInput(Age=55, Sex_M=1, RestingBP=140, Cholesterol=250, MaxHR=150)
    assert patient.Age == 55

    resp = PredictionResponse(
        risk_score=72.0, uncertainty=8.0,
        confidence_level="High", narrative="Test", model_used="unc",
    )
    assert resp.risk_score == 72.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
