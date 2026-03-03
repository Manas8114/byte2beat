"""
UncertaintyML — Uncertainty-Aware Medical Risk Assessment Framework.

An open-source toolkit for building trustworthy clinical ML models
with uncertainty quantification, concept-bottleneck interpretability,
and natural-language explanations.
"""

# --- CRITICAL FIX: sklearn compatibility for Python 3.13+ ---
# Some older dependencies like TabPFN import _is_pandas_df from sklearn.utils.validation
# which was removed in scikit-learn 1.6+.
import sklearn.utils.validation
if not hasattr(sklearn.utils.validation, '_is_pandas_df'):
    try:
        import pandas as pd
        def _is_pandas_df(X):
            return isinstance(X, pd.DataFrame)
        sklearn.utils.validation._is_pandas_df = _is_pandas_df
    except ImportError:
        pass

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
