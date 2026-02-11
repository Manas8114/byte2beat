"""
Example adapter: Pima Indians Diabetes Dataset.

Demonstrates how to extend UncertaintyML to a new disease domain.
Dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

This adapter can be used with any CSV that follows the Pima schema:
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
DiabetesPedigreeFunction, Age, Outcome
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from uncertaintyml.data import DatasetAdapter


class DiabetesAdapter(DatasetAdapter):
    """
    Adapter for the Pima Indians Diabetes dataset.

    Usage:
        adapter = DiabetesAdapter()
        X, y, concept_map = adapter.load("path/to/diabetes.csv")

    The dataset has 768 samples, 8 features, and a binary target (Outcome).
    Zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI
    are treated as missing and imputed with column medians.
    """

    ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    def get_concept_map(self) -> Dict[str, List[str]]:
        return {
            "Metabolic": ["Glucose", "Insulin", "BMI"],
            "Cardiovascular": ["BloodPressure"],
            "Anthropometric": ["SkinThickness", "BMI"],
            "Genetic": ["DiabetesPedigreeFunction"],
            "Demographics": ["Age", "Pregnancies"],
        }

    def load(self, filepath: str, *extra_paths: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
        """
        Load and preprocess the Pima diabetes CSV.

        Handles the well-known issue of zeros as missing values
        in clinical measurement columns.
        """
        df = pd.read_csv(filepath)

        expected_cols = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
        ]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")

        # Replace biologically impossible zeros with NaN, then impute
        for col in self.ZERO_INVALID_COLS:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

        y = df["Outcome"]
        X = df.drop(columns=["Outcome"])

        return X, y, self.get_concept_map()
