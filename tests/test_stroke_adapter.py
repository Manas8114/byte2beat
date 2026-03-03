import pytest
import pandas as pd
import numpy as np
import os
from uncertaintyml.adapters.stroke import StrokeAdapter

@pytest.fixture
def sample_stroke_data(tmp_path):
    # Create synthetic stroke dataset
    n_samples = 10
    
    df = pd.DataFrame({
        'id': range(1, n_samples + 1),
        'gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Other'],
        'age': [30, 45, 60, 25, 75, 40, 55, 65, 35, 50],
        'hypertension': [0, 1, 0, 0, 1, 0, 1, 1, 0, 0],
        'heart_disease': [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        'ever_married': ['Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No'],
        'work_type': ['Private', 'Self-employed', 'Govt_job', 'Private', 'Self-employed', 'Private', 'Govt_job', 'Private', 'Private', 'Private'],
        'Residence_type': ['Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural'],
        'avg_glucose_level': [100.5, 200.0, 150.5, 90.0, 250.0, 110.0, 180.0, 210.0, 105.0, 120.0],
        'bmi': [25.5, 30.0, 'N/A', 22.0, 35.5, 28.0, 32.5, np.nan, 26.0, 29.0],
        'smoking_status': ['formerly smoked', 'never smoked', 'smokes', 'Unknown', 'smokes', 'never smoked', 'formerly smoked', 'smokes', 'Unknown', 'never smoked'],
        'stroke': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    })
    
    file_path = tmp_path / "stroke_test.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

def test_stroke_adapter_load(sample_stroke_data):
    adapter = StrokeAdapter()
    X, y, concept_map = adapter.load(sample_stroke_data)
    
    # 1. Check basic shapes
    # Expecting 9 samples (since 'Other' gender is dropped)
    assert len(X) == 9
    assert len(y) == 9
    
    # 2. Check that ID is dropped
    assert 'id' not in X.columns
    
    # 3. Check BMI imputation
    # Original data had 2 missing/N/A values in the 9 valid rows
    assert not X['bmi'].isnull().any()
    
    # 4. Check dummy variables
    # E.g., gender_Male, gender_Female should exist
    assert 'gender_Male' in X.columns or 'gender_Female' in X.columns
    assert 'ever_married_Yes' in X.columns or 'ever_married_No' in X.columns
    
    # 5. Check output types
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all() # all numeric

def test_stroke_concept_map():
    adapter = StrokeAdapter()
    concept_map = adapter.get_concept_map()
    
    assert isinstance(concept_map, dict)
    assert "Demographics" in concept_map
    assert "Vitals & Metabolic" in concept_map
    assert "avg_glucose_level" in concept_map["Vitals & Metabolic"]
