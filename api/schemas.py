"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class PatientInput(BaseModel):
    """Single patient prediction request."""
    Age: float = Field(..., description="Patient age in years", ge=0, le=120)
    Sex_M: int = Field(0, description="Male=1, Female=0")
    RestingBP: float = Field(120, description="Resting blood pressure (mmHg)")
    Cholesterol: float = Field(200, description="Serum cholesterol (mg/dL)")
    FastingBS: int = Field(0, description="Fasting blood sugar > 120 mg/dL (1=true, 0=false)")
    MaxHR: float = Field(150, description="Maximum heart rate achieved")
    Oldpeak: float = Field(0.0, description="ST depression")
    ExerciseAngina_Y: int = Field(0, description="Exercise-induced angina (1=yes)")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 55,
                "Sex_M": 1,
                "RestingBP": 140,
                "Cholesterol": 250,
                "FastingBS": 0,
                "MaxHR": 150,
                "Oldpeak": 1.2,
                "ExerciseAngina_Y": 0,
            }
        }


class PredictionResponse(BaseModel):
    """Prediction result with uncertainty, narrative, and performance metadata."""
    risk_score: float = Field(..., description="Risk percentage (0-100)")
    uncertainty: float = Field(..., description="Uncertainty percentage")
    confidence_level: str = Field(..., description="High / Medium / Low")
    narrative: str = Field(..., description="Clinical narrative in markdown")
    model_used: str = Field(..., description="Model that produced the prediction")
    cache_hit: bool = Field(False, description="Whether result came from cache")
    route_taken: str = Field("direct", description="Routing path: fast_path / slow_path / direct")


class BatchPredictionResponse(BaseModel):
    """Batch prediction results."""
    predictions: List[PredictionResponse]
    total_patients: int
    avg_risk: float
    avg_uncertainty: float
    high_risk_count: int
    low_confidence_count: int


class ModelInfoResponse(BaseModel):
    """Model metadata."""
    available_models: List[str]
    version: str
    trained_at: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check with cache stats."""
    status: str
    models_loaded: int
    version: str
    cache_stats: Optional[Dict[str, Any]] = None
