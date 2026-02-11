"""
Dataset adapters for loading and preprocessing clinical data.

Provides an abstract DatasetAdapter pattern so new diseases/datasets can be
plugged in without modifying core logic.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class DatasetAdapter(ABC):
    """
    Abstract base for loading any clinical dataset into a unified schema.

    Subclass this to support new diseases/datasets. Every adapter must produce:
    - X: feature DataFrame
    - y: target Series
    - concept_map: dict mapping concept names to feature lists
    """

    @abstractmethod
    def load(self, primary_path: str, *extra_paths: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
        """Load and return (X, y, concept_map)."""
        ...

    @abstractmethod
    def get_concept_map(self) -> Dict[str, List[str]]:
        """Return concept-to-features mapping for interpretability."""
        ...


class HeartDiseaseAdapter(DatasetAdapter):
    """
    Adapter for UCI Heart Disease + Cardiac Failure datasets.
    Produces a unified schema for cardiovascular risk assessment.
    """

    def get_concept_map(self) -> Dict[str, List[str]]:
        return {
            "Demographics": ["Age", "Sex_M", "Sex_F", "Height", "Weight"],
            "Vitals": ["RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Glucose"],
            "Lifestyle": ["Smoke", "Alcohol", "Active"],
            "Clinical": [
                "Oldpeak",
                "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA", "ChestPainType_ASY",
                "RestingECG_Normal", "RestingECG_ST", "RestingECG_LVH",
                "ExerciseAngina_Y",
                "ST_Slope_Flat", "ST_Slope_Up", "ST_Slope_Down",
            ],
        }

    def load(self, processed_path: str, *extra_paths: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
        """
        Load heart disease data from processed CSV, optionally merging
        a secondary dataset for higher sample count.
        """
        df_proc = self._load_heart_processed(processed_path)
        base_path = extra_paths[0] if extra_paths else None

        if base_path:
            try:
                df_base = self._load_cardio_base(base_path)
                all_cols = set(df_proc.columns).union(set(df_base.columns))

                for c in all_cols:
                    if c not in df_proc.columns:
                        df_proc[c] = np.nan
                    if c not in df_base.columns:
                        df_base[c] = np.nan

                df_final = pd.concat([df_proc, df_base], axis=0, ignore_index=True)

                for col in df_final.columns:
                    if df_final[col].dtype in ["float64", "int64"]:
                        df_final[col] = df_final[col].fillna(df_final[col].median())
                    else:
                        mode = df_final[col].mode()
                        df_final[col] = df_final[col].fillna(mode.iloc[0] if not mode.empty else 0)
            except Exception as e:
                print(f"Warning: Could not merge secondary dataset: {e}")
                df_final = df_proc
        else:
            df_final = df_proc

        if "HeartDisease" not in df_final.columns:
            raise ValueError("Target column 'HeartDisease' missing from dataset.")

        y = df_final["HeartDisease"]
        X = df_final.drop(columns=["HeartDisease"])
        return X, y, self.get_concept_map()

    @staticmethod
    def _load_heart_processed(filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        for col in df.columns:
            if df[col].dtype == "bool" or df[col].astype(str).isin(["True", "False"]).any():
                df[col] = df[col].map({"True": 1, True: 1, "False": 0, False: 0})
        return df

    @staticmethod
    def _load_cardio_base(filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, sep=";")
        df["Age"] = (df["age"] / 365.25).astype(int)
        df["Sex_M"] = (df["gender"] == 2).astype(int)
        df = df.rename(columns={
            "ap_hi": "RestingBP",
            "gluc": "Glucose",
            "smoke": "Smoke",
            "alco": "Alcohol",
            "active": "Active",
            "cardio": "HeartDisease",
            "height": "Height",
        })
        df["Cholesterol"] = df["cholesterol"].map({1: 180, 2: 225, 3: 260})
        df = df.drop(columns=["id", "age", "gender", "ap_lo", "cholesterol"])
        return df


# Backward-compatible function
def load_and_preprocess_data(processed_path: str, base_path: str = None):
    """Legacy wrapper. Use HeartDiseaseAdapter instead."""
    adapter = HeartDiseaseAdapter()
    if base_path:
        return adapter.load(processed_path, base_path)
    return adapter.load(processed_path)


def get_concept_map():
    """Legacy wrapper. Use HeartDiseaseAdapter().get_concept_map() instead."""
    return HeartDiseaseAdapter().get_concept_map()
