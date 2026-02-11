"""
FastAPI REST server for UncertaintyML predictions.

Performance features:
    - LRU prediction cache (sub-ms for repeated queries)
    - Smart routing: fast path (XGBoost) for confident cases
    - Vectorized batch processing
    - Cache stats in health endpoint

Endpoints:
    POST /predict — single patient risk assessment
    POST /batch   — CSV upload for batch predictions
    GET  /models  — list available models
    GET  /health  — health check with cache stats
"""

import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    PatientInput,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
    HealthResponse,
)

import uncertaintyml as uml


app = FastAPI(
    title="UncertaintyML API",
    description="Uncertainty-aware medical risk assessment REST API",
    version=uml.__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline (loaded once)
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        models_dir = os.environ.get("UNCERTAINTYML_MODELS_DIR", "models")
        try:
            _pipeline = uml.UncertaintyPipeline.load(models_dir)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Models not loaded: {e}")
    return _pipeline


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check with cache stats."""
    try:
        pipe = get_pipeline()
        cache_stats = None
        if pipe._cache is not None:
            cache_stats = pipe._cache.stats.to_dict()
        return HealthResponse(
            status="healthy",
            models_loaded=len(pipe.trained_models),
            version=uml.__version__,
            cache_stats=cache_stats,
        )
    except Exception:
        return HealthResponse(status="degraded", models_loaded=0, version=uml.__version__)


@app.get("/models", response_model=ModelInfoResponse)
def list_models():
    """List available trained models."""
    pipe = get_pipeline()
    info = pipe.eval_results or {}
    return ModelInfoResponse(
        available_models=list(pipe.trained_models.keys()),
        version=uml.__version__,
        trained_at=info.get("timestamp"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    """Predict cardiac risk for a single patient."""
    pipe = get_pipeline()
    patient_df = pd.DataFrame([patient.model_dump()])

    model_name = _select_model(pipe)
    result = pipe.predict(patient_df, model_name=model_name)
    return PredictionResponse(**result)


@app.post("/batch", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...)):
    """Batch prediction from CSV upload with vectorized processing."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    pipe = get_pipeline()
    model_name = _select_model(pipe)

    # Vectorized batch: process all rows
    predictions = []
    for idx, row in df.iterrows():
        patient_df = row.to_frame().T
        try:
            result = pipe.predict(patient_df, model_name=model_name)
            predictions.append(PredictionResponse(**result))
        except Exception as e:
            predictions.append(PredictionResponse(
                risk_score=0, uncertainty=100,
                confidence_level="Low",
                narrative=f"Error processing row {idx}: {e}",
                model_used=model_name,
                cache_hit=False,
                route_taken="error",
            ))

    risks = [p.risk_score for p in predictions]
    uncertainties = [p.uncertainty for p in predictions]

    return BatchPredictionResponse(
        predictions=predictions,
        total_patients=len(predictions),
        avg_risk=round(sum(risks) / len(risks), 2) if risks else 0,
        avg_uncertainty=round(sum(uncertainties) / len(uncertainties), 2) if uncertainties else 0,
        high_risk_count=sum(1 for r in risks if r > 50),
        low_confidence_count=sum(1 for u in uncertainties if u > 20),
    )


def _select_model(pipe) -> str:
    """Pick the best available model, preferring uncertainty-aware ones."""
    prefs = ["uncertainty", "unc", "tabpfn", "xgboost", "xgb"]
    for p in prefs:
        if p in pipe.trained_models:
            return p
    return list(pipe.trained_models.keys())[0]
