import pandas as pd
import numpy as np

def generate_clinical_narrative(patient_data: pd.Series, risk_score: float, uncertainty_score: float, shap_values: dict = None):
    """
    Generates a natural-language clinical narrative based on patient data and model outputs.
    Currently mimics an LLM response using sophisticated templates (Option B - LLM Augmented).
    
    Args:
        patient_data (pd.Series): The raw feature values for the patient.
        risk_score (float): The predicted risk percentage (0-100).
        uncertainty_score (float): The uncertainty percentage/score.
        shap_values (dict): Optional dictionary of feature importance {feature_name: value}.
    
    Returns:
        str: A Markdown-formatted clinical narrative.
    """
    
    # 1. Context Analysis
    risk_level = "High" if risk_score > 50 else "Low"
    conf_level = "High" if uncertainty_score < 10 else ("Medium" if uncertainty_score < 20 else "Low")
    
    # 2. Key Driver Identification (if SHAP provided, otherwise heuristic)
    # Simple heuristic fallback if SHAP is missing
    drivers = []
    if shap_values:
        sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        top_3 = sorted_shap[:3]
        for k, v in top_3:
             drivers.append(f"**{k.replace('_', ' ').title()}**")
    else:
        # Fallback to checking abnormal values (heuristic)
        if patient_data.get('Cholesterol', 0) > 200: drivers.append("**Cholesterol**")
        if patient_data.get('RestingBP', 0) > 130: drivers.append("**Blood Pressure**")
        if patient_data.get('MaxHR', 0) < 100: drivers.append("**Low Max HR**")
        if patient_data.get('Age', 0) > 55: drivers.append("**Age**")
        if patient_data.get('FastingBS', 0) > 0: drivers.append("**Blood Sugar**")

    driver_text = " and ".join(drivers) if drivers else "general profile"

    # 3. Narrative Construction (Template System to mimic LLM)
    
    # Template A: High Risk, High Uncertainty (The "Warning" Case)
    if risk_level == "High" and conf_level == "Low":
        return f"""
### ⚠️ AI Assessment: Complex Case
The model estimates a **{risk_score:.1f}% risk** (High), but strictly flags this as a **Low Confidence** prediction ({uncertainty_score:.1f}% uncertainty).

**Clinical Context:**
This patient presents an unusual combination of factors, deviating from the typical training distribution. While {driver_text} appear to be driving the risk, the high uncertainty suggests these signals might be confounded.

**Recommendation:**
Standard guidelines may not fully apply. **Manual clinical review is strongly recommended** before intervention.
"""

    # Template B: High Risk, High Confidence (The "Clear" Case)
    elif risk_level == "High":
        return f"""
### 🚨 AI Assessment: Elevated Risk
The model identifies a **{risk_score:.1f}% risk** of a cardiac event with **{conf_level} Confidence**.

**Key Drivers:**
The risk is primarily driven by {driver_text}. The patient's profile strongly aligns with high-risk clusters in the population data.

**Recommendation:**
Immediate prophylactic intervention targeting {drivers[0] if drivers else "the relevant risk factors"} is advised.
"""

    # Template C: Low Risk (The "Healthy" Case)
    else:
        return f"""
### ✅ AI Assessment: Low Risk
The patient shows a **{risk_score:.1f}% risk** profile, consistent with valid healthy baselines (**{conf_level} Confidence**).

**Analysis:**
Vitals such as {driver_text} remain within nominal ranges. No specific intervention is currently flagged.
"""
