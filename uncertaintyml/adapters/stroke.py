"""
Adapter for the Stroke Prediction Dataset.
Dataset: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from uncertaintyml.data import DatasetAdapter

class StrokeAdapter(DatasetAdapter):
    """
    Adapter for the Stroke Prediction dataset.

    Usage:
        adapter = StrokeAdapter()
        X, y, concept_map = adapter.load("path/to/healthcare-dataset-stroke-data.csv")
    """

    def get_concept_map(self) -> Dict[str, List[str]]:
        return {
            "Demographics": ["age", "gender_Male", "gender_Female", "ever_married_Yes", "Residence_type_Urban"],
            "Vitals & Metabolic": ["avg_glucose_level", "bmi"],
            "Clinical History": ["hypertension", "heart_disease"],
            "Lifestyle": ["smoking_status_formerly smoked", "smoking_status_never smoked", "smoking_status_smokes"],
            "Work": ["work_type_Private", "work_type_Self-employed", "work_type_Govt_job", "work_type_children"]
        }

    def load(self, filepath: str, *extra_paths: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
        """Load and preprocess the Stroke CSV."""
        df = pd.read_csv(filepath)

        # Handle missing BMI
        df['bmi'] = df['bmi'].replace('N/A', np.nan)
        df['bmi'] = df['bmi'].astype(float)
        df['bmi'] = df['bmi'].fillna(df['bmi'].median())
        
        # Drop ID
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            
        # Drop 'Other' gender as it's typically highly imbalanced/rare in this dataset
        df = df[df['gender'] != 'Other']
        
        # Convert categorical to dummies
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
        
        y = df["stroke"]
        X = df.drop(columns=["stroke"])
        
        # Feature engineering
        X['risk_composite'] = (
            X.get('hypertension', 0) + 
            X.get('heart_disease', 0) + 
            (X['age'] > 65).astype(float)
        )
        X['glucose_bmi'] = X.get('avg_glucose_level', 0) * X.get('bmi', 0) / 1000.0
        
        # Force float64 for SHAP
        for col in X.columns:
            X[col] = X[col].astype(np.float64)

        return X, y, self.get_concept_map()
