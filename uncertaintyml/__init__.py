"""
UncertaintyML — Uncertainty-Aware Medical Risk Assessment Framework.

An open-source toolkit for building trustworthy clinical ML models
with uncertainty quantification, concept-bottleneck interpretability,
and natural-language explanations.
"""

__version__ = "0.1.0"

from uncertaintyml.models import UncertaintyModel, ModelRegistry
from uncertaintyml.pipeline import UncertaintyPipeline, PipelineConfig
from uncertaintyml.interpret import ConceptBottleneck, NarrativeEngine
from uncertaintyml.data import DatasetAdapter, HeartDiseaseAdapter
from uncertaintyml.cache import PredictionCache
from uncertaintyml.adapters import DiabetesAdapter

__all__ = [
    "UncertaintyModel",
    "ModelRegistry",
    "UncertaintyPipeline",
    "PipelineConfig",
    "ConceptBottleneck",
    "NarrativeEngine",
    "DatasetAdapter",
    "HeartDiseaseAdapter",
    "PredictionCache",
    "DiabetesAdapter",
]
