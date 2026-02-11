"""
Unit Tests for Models.
Verifies core functionality of XGBoost, TabPFN, and Uncertainty models.
"""
import pytest
import sys
import os
import numpy as np
import pandas as pd
import joblib

# Add project root and src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from uncertaintyml.models import UncertaintyModel, get_xgboost
except ImportError:
    from utils_model import UncertaintyModel, get_xgboost


class TestXGBoostModel:
    """Tests for XGBoost baseline model."""
    
    @pytest.fixture
    def trained_model(self):
        """Load pre-trained XGBoost model."""
        model_path = 'models/xgb_model.pkl'
        if not os.path.exists(model_path):
            pytest.skip("XGBoost model not found. Run run_experiment.py first.")
        return joblib.load(model_path)
    
    @pytest.fixture
    def test_data(self):
        """Load test data."""
        X_path = 'models/X_test.pkl'
        if not os.path.exists(X_path):
            pytest.skip("Test data not found. Run run_experiment.py first.")
        return joblib.load(X_path)
    
    def test_predict_returns_correct_shape(self, trained_model, test_data):
        """Test that predictions have correct shape."""
        predictions = trained_model.predict(test_data)
        assert predictions.shape[0] == test_data.shape[0]
    
    def test_predict_proba_returns_probabilities(self, trained_model, test_data):
        """Test that predict_proba returns valid probabilities."""
        proba = trained_model.predict_proba(test_data)
        
        # Should have 2 columns (binary classification)
        assert proba.shape[1] == 2
        
        # Probabilities should sum to 1
        row_sums = proba.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(row_sums)))
        
        # All values should be between 0 and 1
        assert np.all(proba >= 0) and np.all(proba <= 1)
    
    def test_feature_importance_available(self, trained_model, test_data):
        """Test that feature importance is available."""
        importances = trained_model.feature_importances_
        assert len(importances) == test_data.shape[1]
        assert np.all(importances >= 0)


class TestUncertaintyModel:
    """Tests for MC Dropout Uncertainty Model."""
    
    @pytest.fixture
    def trained_model(self):
        """Load pre-trained uncertainty model."""
        model_path = 'models/uncertainty_model.pkl'
        if not os.path.exists(model_path):
            pytest.skip("Uncertainty model not found. Run run_experiment.py first.")
        return joblib.load(model_path)
    
    @pytest.fixture
    def test_data(self):
        """Load test data."""
        X_path = 'models/X_test.pkl'
        if not os.path.exists(X_path):
            pytest.skip("Test data not found. Run run_experiment.py first.")
        return joblib.load(X_path)
    
    def test_predict_uncertainty_returns_two_arrays(self, trained_model, test_data):
        """Test that predict_uncertainty returns mean and std."""
        sample = test_data.iloc[:10]
        mean_preds, std_preds = trained_model.predict_uncertainty(sample, n_samples=10)
        
        assert len(mean_preds) == 10
        assert len(std_preds) == 10
    
    def test_uncertainty_is_positive(self, trained_model, test_data):
        """Test that uncertainty (std) is always positive."""
        sample = test_data.iloc[:10]
        _, std_preds = trained_model.predict_uncertainty(sample, n_samples=10)
        
        assert np.all(std_preds >= 0)
    
    def test_mean_predictions_are_probabilities(self, trained_model, test_data):
        """Test that mean predictions are valid probabilities."""
        sample = test_data.iloc[:10]
        mean_preds, _ = trained_model.predict_uncertainty(sample, n_samples=10)
        
        assert np.all(mean_preds >= 0) and np.all(mean_preds <= 1)
    
    def test_scaler_is_fitted(self, trained_model):
        """Test that the internal scaler is fitted."""
        assert trained_model.scaler is not None
        assert hasattr(trained_model.scaler, 'mean_')  # Fitted scaler has mean_
    
    def test_high_uncertainty_on_garbage_data(self, trained_model, test_data):
        """Test that garbage data produces high uncertainty."""
        # Create garbage data
        n_garbage = 5
        garbage_df = pd.DataFrame(
            np.random.uniform(-500, 500, size=(n_garbage, test_data.shape[1])),
            columns=test_data.columns
        )
        
        _, std_preds = trained_model.predict_uncertainty(garbage_df, n_samples=30)
        avg_uncertainty = np.mean(std_preds)
        
        # Garbage data should have higher than 5% uncertainty
        assert avg_uncertainty > 0.05, f"Uncertainty too low on garbage data: {avg_uncertainty}"


class TestModelConsistency:
    """Tests for model consistency and integration."""
    
    def test_models_exist(self):
        """Test that all required model files exist."""
        required_files = [
            'models/xgb_model.pkl',
            'models/uncertainty_model.pkl',
            'models/X_test.pkl',
            'models/y_test.pkl'
        ]
        
        for f in required_files:
            assert os.path.exists(f), f"Missing required file: {f}"
    
    def test_test_data_shape_matches_models(self):
        """Test that test data shape matches model expectations."""
        X_test = joblib.load('models/X_test.pkl')
        y_test = joblib.load('models/y_test.pkl')
        
        assert len(X_test) == len(y_test), "X_test and y_test length mismatch"
        assert X_test.shape[1] > 0, "X_test has no features"


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
