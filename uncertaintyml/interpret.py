"""
Interpretability: Concept-Bottleneck mapping + Narrative generation engine.

Combines SHAP feature importance aggregation with clinician-friendly
natural-language explanations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class ConceptBottleneck:
    """
    Maps raw feature importances (e.g. SHAP values) into higher-level
    clinical concepts. Bridges ML explainability and clinical reasoning.

    Usage:
        cb = ConceptBottleneck(concept_map)
        concept_scores = cb.aggregate(shap_values_dict)
        # {"Vitals": 0.72, "Demographics": 0.15, ...}
    """

    def __init__(self, concept_map: Dict[str, List[str]]):
        self.concept_map = concept_map

    def aggregate(self, feature_importances: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate raw feature importances into concept-level scores.

        Args:
            feature_importances: {feature_name: importance_value}

        Returns:
            {concept_name: aggregated_importance}
        """
        concept_scores = {}
        for concept, features in self.concept_map.items():
            scores = [abs(feature_importances.get(f, 0.0)) for f in features]
            concept_scores[concept] = sum(scores)

        total = sum(concept_scores.values())
        if total > 0:
            concept_scores = {k: v / total for k, v in concept_scores.items()}
        return concept_scores

    def get_dominant_concept(self, feature_importances: Dict[str, float]) -> str:
        """Return the concept with highest aggregated importance."""
        scores = self.aggregate(feature_importances)
        return max(scores, key=scores.get) if scores else "Unknown"

    def get_narrative_drivers(self, feature_importances: Dict[str, float], top_n: int = 3) -> List[str]:
        """Return top-N contributing concepts sorted by importance."""
        scores = self.aggregate(feature_importances)
        sorted_concepts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [c[0] for c in sorted_concepts[:top_n]]


class NarrativeEngine:
    """
    Generates natural-language clinical narratives from model outputs.
    Supports pluggable templates for different clinical contexts.
    """

    def __init__(self, concept_bottleneck: ConceptBottleneck = None):
        self.concept_bottleneck = concept_bottleneck

    def generate(
        self,
        patient_data: pd.Series,
        risk_score: float,
        uncertainty_score: float,
        shap_values: Dict[str, float] = None,
    ) -> str:
        """
        Generate a markdown-formatted clinical narrative.

        Args:
            patient_data: Raw feature values for the patient
            risk_score: Predicted risk percentage (0-100)
            uncertainty_score: Uncertainty percentage
            shap_values: Optional feature importance dict
        """
        risk_level = "High" if risk_score > 50 else "Low"
        conf_level = "High" if uncertainty_score < 10 else ("Medium" if uncertainty_score < 20 else "Low")

        drivers = self._identify_drivers(patient_data, shap_values)
        driver_text = " and ".join(drivers) if drivers else "general profile"

        # Concept-level insight
        concept_insight = ""
        if self.concept_bottleneck and shap_values:
            scores = self.concept_bottleneck.aggregate(shap_values)
            top_concept = max(scores, key=scores.get) if scores else None
            if top_concept:
                pct = scores[top_concept] * 100
                concept_insight = f"\n**Concept Analysis:** Risk primarily driven by **{top_concept}** ({pct:.0f}% contribution).\n"

        if risk_level == "High" and conf_level == "Low":
            return self._template_high_risk_low_confidence(risk_score, uncertainty_score, driver_text, concept_insight)
        elif risk_level == "High":
            return self._template_high_risk(risk_score, conf_level, driver_text, drivers, concept_insight)
        else:
            return self._template_low_risk(risk_score, conf_level, driver_text, concept_insight)

    @staticmethod
    def _identify_drivers(patient_data: pd.Series, shap_values: Dict = None) -> List[str]:
        drivers = []
        if shap_values:
            sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
            drivers = [f"**{k.replace('_', ' ').title()}**" for k, _ in sorted_shap[:3]]
        else:
            if patient_data.get("Cholesterol", 0) > 200:
                drivers.append("**Cholesterol**")
            if patient_data.get("RestingBP", 0) > 130:
                drivers.append("**Blood Pressure**")
            if patient_data.get("MaxHR", 0) < 100:
                drivers.append("**Low Max HR**")
            if patient_data.get("Age", 0) > 55:
                drivers.append("**Age**")
            if patient_data.get("FastingBS", 0) > 0:
                drivers.append("**Blood Sugar**")
        return drivers

    @staticmethod
    def _template_high_risk_low_confidence(risk, uncertainty, driver_text, concept_insight):
        return f"""
### ⚠️ AI Assessment: Complex Case
The model estimates a **{risk:.1f}% risk** (High), but flags this as a **Low Confidence** prediction ({uncertainty:.1f}% uncertainty).

**Clinical Context:**
This patient presents an unusual combination of factors. While {driver_text} appear to be driving the risk, the high uncertainty suggests these signals might be confounded.
{concept_insight}
**Recommendation:**
Standard guidelines may not fully apply. **Manual clinical review is strongly recommended** before intervention.
"""

    @staticmethod
    def _template_high_risk(risk, conf_level, driver_text, drivers, concept_insight):
        return f"""
### 🚨 AI Assessment: Elevated Risk
The model identifies a **{risk:.1f}% risk** of a cardiac event with **{conf_level} Confidence**.

**Key Drivers:**
The risk is primarily driven by {driver_text}.
{concept_insight}
**Recommendation:**
Immediate prophylactic intervention targeting {drivers[0] if drivers else "the relevant risk factors"} is advised.
"""

    @staticmethod
    def _template_low_risk(risk, conf_level, driver_text, concept_insight):
        return f"""
### ✅ AI Assessment: Low Risk
The patient shows a **{risk:.1f}% risk** profile, consistent with healthy baselines (**{conf_level} Confidence**).

**Analysis:**
Vitals such as {driver_text} remain within nominal ranges.
{concept_insight}
"""


# Backward-compatible function
def generate_clinical_narrative(patient_data, risk_score, uncertainty_score, shap_values=None):
    """Legacy wrapper. Use NarrativeEngine instead."""
    engine = NarrativeEngine()
    return engine.generate(patient_data, risk_score, uncertainty_score, shap_values)
