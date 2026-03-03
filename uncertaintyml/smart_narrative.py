"""
SmartNarrativeEngine — Advanced clinical narrative with counterfactual reasoning.
No external API keys required. Uses rule-based logic and model re-inference.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class SmartNarrativeEngine:
    """
    Generates rich clinical narratives with counterfactual "what-if" analysis
    by re-running the model with modified patient features.
    """

    # Clinical thresholds for common features
    THRESHOLDS = {
        'age': {'label': 'Age', 'unit': 'yrs', 'high': 55},
        'cholesterol': {'label': 'Cholesterol', 'unit': 'mg/dL', 'high': 200, 'target': 180},
        'restingbp': {'label': 'Blood Pressure', 'unit': 'mmHg', 'high': 130, 'target': 120},
        'maxhr': {'label': 'Max Heart Rate', 'unit': 'bpm', 'low': 100},
        'fastingbs': {'label': 'Fasting Blood Sugar', 'unit': '', 'high': 0},
        'bmi': {'label': 'BMI', 'unit': 'kg/m²', 'high': 30, 'target': 25},
        'avg_glucose_level': {'label': 'Avg Glucose', 'unit': 'mg/dL', 'high': 140, 'target': 100},
        'hypertension': {'label': 'Hypertension', 'unit': '', 'high': 0, 'target': 0},
        'heart_disease': {'label': 'Heart Disease', 'unit': '', 'high': 0, 'target': 0},
    }

    def __init__(self, model=None, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names or []

    def _get_risk(self, input_data: pd.DataFrame) -> float:
        """Get risk score from the model."""
        if self.model is None:
            return 0.0
        try:
            if hasattr(self.model, 'predict_uncertainty'):
                means, _ = self.model.predict_uncertainty(input_data, n_samples=20)
                return float(means[0]) * 100
            elif hasattr(self.model, 'predict_proba'):
                return float(self.model.predict_proba(input_data)[0][1]) * 100
            else:
                return float(self.model.predict(input_data)[0]) * 100
        except Exception:
            return 0.0

    def _find_modifiable_factors(self, patient_data: pd.Series) -> List[Dict]:
        """Identify modifiable risk factors based on clinical thresholds."""
        factors = []
        for key, info in self.THRESHOLDS.items():
            for col in patient_data.index:
                if key in col.lower():
                    val = float(patient_data[col])
                    if 'high' in info and val > info['high']:
                        factors.append({
                            'feature': col,
                            'label': info['label'],
                            'current': val,
                            'target': info.get('target', info['high']),
                            'unit': info['unit'],
                            'direction': 'reduce'
                        })
                    elif 'low' in info and val < info['low']:
                        factors.append({
                            'feature': col,
                            'label': info['label'],
                            'current': val,
                            'target': info['low'],
                            'unit': info['unit'],
                            'direction': 'increase'
                        })
                    break
        return factors

    def _run_counterfactual(self, input_data: pd.DataFrame, feature: str, target_val: float) -> float:
        """Re-run the model with one feature changed to get counterfactual risk."""
        cf_data = input_data.copy()
        cf_data[feature] = target_val
        return self._get_risk(cf_data)

    def generate(
        self,
        patient_data: pd.Series,
        risk_score: float,
        uncertainty_score: float,
        input_data: pd.DataFrame = None,
        shap_values: Dict[str, float] = None,
        disease_type: str = "Cardiovascular"
    ) -> str:
        """Generate a rich clinical narrative with counterfactual reasoning."""
        risk_level = "High" if risk_score > 50 else "Moderate" if risk_score > 30 else "Low"
        conf_level = "High" if uncertainty_score < 10 else ("Medium" if uncertainty_score < 20 else "Low")

        # Identify top SHAP drivers or fallback
        drivers = self._get_drivers(patient_data, shap_values)
        driver_text = ", ".join(drivers[:3]) if drivers else "general clinical profile"

        # Modifiable factors
        modifiable = self._find_modifiable_factors(patient_data)

        # Counterfactual analysis
        counterfactuals = []
        if input_data is not None and self.model is not None and modifiable:
            for factor in modifiable[:3]:
                cf_risk = self._run_counterfactual(input_data, factor['feature'], factor['target'])
                delta = risk_score - cf_risk
                if abs(delta) > 0.5:
                    counterfactuals.append({
                        **factor,
                        'new_risk': cf_risk,
                        'delta': delta
                    })

        # Build Data-Driven Insights
        sections = []

        # Header
        emoji = "🚨" if risk_level == "High" else "⚠️" if risk_level == "Moderate" else "✅"
        sections.append(f"### {emoji} Quantitative Risk Analysis")
        sections.append("")
        
        # Uncertainty
        if conf_level == "Low":
            sections.append(
                f"> ⚠️ **Low Confidence Prediction:** Model is exhibiting high variance (±{uncertainty_score:.1f}%) "
                "on this input vector. Interpret risk score with caution."
            )
            sections.append("")

        # Key drivers
        if drivers:
            sections.append("**Top Mathematical Risk Drivers (SHAP/Feature Weights):**")
            for d in drivers[:4]:
                sections.append(f"- {d}")
            sections.append("")

        # Counterfactual reasoning (True Model Inference Runs)
        if counterfactuals:
            sections.append("**🔬 Counterfactual Analysis (Re-inference deltas):**")
            sections.append("")
            sections.append("| Feature Adjusted | Current Value | Target Value | Model Risk Δ |")
            sections.append("|---|---|---|---|")
            for cf in sorted(counterfactuals, key=lambda x: x['delta'], reverse=True):
                arrow = "↓" if cf['delta'] > 0 else "↑"
                sections.append(
                    f"| {cf['direction'].title()} {cf['label']} | "
                    f"{cf['current']:.0f} {cf['unit']} | "
                    f"{cf['target']:.0f} {cf['unit']} | "
                    f"**{cf['delta']:+.1f}%** {arrow} |"
                )
            sections.append("")

            best = max(counterfactuals, key=lambda x: x['delta'])
            sections.append(
                f"💡 **Highest Impact Intervention (Calculated):** Target **{best['label']}** "
                f"({best['current']:.0f} → {best['target']:.0f} {best['unit']}). "
                f"Expected Risk Reduction: **{best['delta']:.1f}%**."
            )

        return "\n".join(sections)

    @staticmethod
    def _get_drivers(patient_data: pd.Series, shap_values: Dict = None) -> List[str]:
        """Extract top risk drivers from SHAP or heuristics."""
        drivers = []
        if shap_values:
            sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
            for k, v in sorted_shap[:4]:
                direction = "↑" if v > 0 else "↓"
                drivers.append(f"**{k.replace('_', ' ').title()}** ({direction} risk)")
        else:
            # Heuristic fallback
            checks = [
                ('Cholesterol', 200, 'Elevated Cholesterol'),
                ('RestingBP', 130, 'Elevated Blood Pressure'),
                ('Age', 55, 'Advanced Age'),
                ('bmi', 30, 'Elevated BMI'),
                ('avg_glucose_level', 140, 'High Glucose'),
                ('hypertension', 0, 'Hypertension History'),
                ('heart_disease', 0, 'Heart Disease History'),
            ]
            for feat, thresh, label in checks:
                val = patient_data.get(feat, None)
                if val is not None:
                    try:
                        if float(val) > thresh:
                            drivers.append(f"**{label}** ({float(val):.0f})")
                    except (ValueError, TypeError):
                        pass
            # Low HR special case
            max_hr = patient_data.get('MaxHR', None)
            if max_hr is not None:
                try:
                    if float(max_hr) < 100:
                        drivers.append(f"**Low Max Heart Rate** ({float(max_hr):.0f} bpm)")
                except (ValueError, TypeError):
                    pass
        return drivers
