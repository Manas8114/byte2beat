"""
Integration Tests for Dashboard.
Verifies dashboard components work correctly.
"""
import pytest
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


class TestNarrativeGenerator:
    """Tests for clinical narrative generation."""
    
    def test_narrative_returns_string(self):
        """Test that narrative generator returns a string."""
        from narrative_generator import generate_clinical_narrative
        import pandas as pd
        
        # Create mock patient data
        patient_data = pd.Series({
            'Age': 55,
            'Cholesterol': 240,
            'RestingBP': 140,
            'MaxHR': 150,
            'FastingBS': 1
        })
        
        narrative = generate_clinical_narrative(patient_data, 65.0, 12.0)
        
        assert isinstance(narrative, str)
        assert len(narrative) > 0
    
    def test_high_risk_high_uncertainty_narrative(self):
        """Test narrative for high risk, low confidence case."""
        from narrative_generator import generate_clinical_narrative
        import pandas as pd
        
        patient_data = pd.Series({
            'Age': 70,
            'Cholesterol': 300,
            'RestingBP': 160,
            'MaxHR': 100,
            'FastingBS': 1
        })
        
        narrative = generate_clinical_narrative(patient_data, 75.0, 25.0)
        
        # Should mention uncertainty/complex case
        assert 'Complex' in narrative or 'Low Confidence' in narrative or 'uncertainty' in narrative.lower()
    
    def test_low_risk_narrative(self):
        """Test narrative for low risk case."""
        from narrative_generator import generate_clinical_narrative
        import pandas as pd
        
        patient_data = pd.Series({
            'Age': 35,
            'Cholesterol': 180,
            'RestingBP': 110,
            'MaxHR': 180,
            'FastingBS': 0
        })
        
        narrative = generate_clinical_narrative(patient_data, 15.0, 5.0)
        
        # Should mention low risk
        assert 'Low Risk' in narrative


class TestEvaluationModule:
    """Tests for evaluation utilities."""
    
    def test_compute_ece(self):
        """Test ECE computation."""
        from evaluation import compute_ece
        import numpy as np
        
        # Perfect calibration: predicted 0.5 for all, 50% actual positive
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.5] * 10)
        
        ece = compute_ece(y_true, y_prob, n_bins=5)
        
        # ECE should be 0 for perfect calibration
        assert ece == 0.0
    
    def test_compute_calibration_data(self):
        """Test calibration data computation."""
        from evaluation import compute_calibration_data
        import numpy as np
        
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.85, 0.95, 0.15, 0.25])
        
        cal_data = compute_calibration_data(y_true, y_prob)
        
        assert 'ece' in cal_data
        assert 'brier_score' in cal_data
        assert 'fraction_of_positives' in cal_data
        assert 'mean_predicted_value' in cal_data
    
    def test_bootstrap_metric(self):
        """Test bootstrap confidence interval computation."""
        from evaluation import bootstrap_metric
        from sklearn.metrics import roc_auc_score
        import numpy as np
        
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.85, 0.95, 0.15, 0.25])
        
        result = bootstrap_metric(y_true, y_prob, roc_auc_score, n_bootstrap=100)
        
        assert 'mean' in result
        assert 'lower' in result
        assert 'upper' in result
        assert result['lower'] <= result['mean'] <= result['upper']


class TestEvalMetrics:
    """Tests for saved evaluation metrics."""
    
    def test_eval_metrics_file_exists(self):
        """Test that eval_metrics.json exists after training."""
        filepath = 'models/eval_metrics.json'
        if not os.path.exists(filepath):
            pytest.skip("eval_metrics.json not found. Run run_experiment.py first.")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert 'models' in data
        assert 'timestamp' in data
    
    def test_model_info_file_exists(self):
        """Test that model_info.json exists after training."""
        filepath = 'models/model_info.json'
        if not os.path.exists(filepath):
            pytest.skip("model_info.json not found. Run run_experiment.py first.")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert 'version' in data
        assert 'trained_at' in data


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
