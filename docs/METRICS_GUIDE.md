# 🫀 Cardiovascular Health Metrics Guide

This document explains each biomarker tracked by the Cardiac Risk Assessment system, including optimal ranges and clinical significance.

---

## Blood Pressure (BP)

### What It Measures

Blood pressure measures the force of blood against artery walls during heart contractions (systolic) and rest (diastolic).

### Values

| Category | Systolic (mmHg) | Diastolic (mmHg) |
|----------|-----------------|------------------|
| **Optimal** | < 120 | < 80 |
| **Elevated** | 120-129 | < 80 |
| **Stage 1 Hypertension** | 130-139 | 80-89 |
| **Stage 2 Hypertension** | ≥ 140 | ≥ 90 |
| **Hypertensive Crisis** | > 180 | > 120 |

### Clinical Significance

- High BP (hypertension) is a major risk factor for heart disease, stroke, and kidney failure
- Often called the "silent killer" because it typically has no symptoms
- **Target: 120/80 mmHg** is considered optimal

---

## Cholesterol

### What It Measures

Total cholesterol measures fats in the blood, including LDL ("bad") and HDL ("good") cholesterol.

### Values

| Category | Total Cholesterol (mg/dL) |
|----------|---------------------------|
| **Desirable** | < 200 |
| **Borderline High** | 200-239 |
| **High** | ≥ 240 |

### LDL Cholesterol (Bad)

| Category | LDL (mg/dL) |
|----------|-------------|
| **Optimal** | < 100 |
| **Near Optimal** | 100-129 |
| **Borderline High** | 130-159 |
| **High** | ≥ 160 |

### HDL Cholesterol (Good)

| Category | HDL (mg/dL) |
|----------|-------------|
| **Low (Risk Factor)** | < 40 (men), < 50 (women) |
| **Optimal** | ≥ 60 |

### Clinical Significance

- High LDL leads to plaque buildup in arteries (atherosclerosis)
- High HDL helps remove cholesterol from arteries
- **Target: Total < 200, LDL < 100, HDL > 60**

---

## Resting Heart Rate (RestingBP vs. MaxHR)

### Resting Heart Rate

| Category | Beats Per Minute (BPM) |
|----------|------------------------|
| **Well-Trained Athletes** | 40-60 |
| **Excellent** | 60-65 |
| **Good** | 66-69 |
| **Average** | 70-72 |
| **Below Average** | 73-78 |
| **Poor** | > 78 |

### Maximum Heart Rate (MaxHR)

- Estimated by: **220 - Age**
- Example: 40-year-old → MaxHR ≈ 180 BPM
- Used for exercise intensity zones

### Clinical Significance

- Lower resting heart rate = more efficient heart function
- Very high or very low MaxHR during stress tests can indicate heart problems
- **Target: Resting HR 60-70 BPM**

---

## Fasting Blood Sugar (FastingBS)

### What It Measures

Glucose level in blood after 8+ hours of fasting.

### Values

| Category | Fasting Glucose (mg/dL) |
|----------|-------------------------|
| **Normal** | < 100 |
| **Prediabetes** | 100-125 |
| **Diabetes** | ≥ 126 |

### Clinical Significance

- Diabetes is a major risk factor for heart disease (2-4x higher risk)
- High blood sugar damages blood vessels over time
- **Target: < 100 mg/dL fasting**

---

## Age

### Risk Factor

| Age Range | Risk Level |
|-----------|------------|
| < 45 (men), < 55 (women) | Lower risk |
| 45-64 (men), 55-64 (women) | Moderate risk |
| ≥ 65 | Higher risk |

### Clinical Significance

- Heart disease risk increases with age
- Men have higher risk earlier; women's risk increases post-menopause
- Non-modifiable risk factor

---

## ST Depression (Oldpeak)

### What It Measures

ST segment depression on ECG during exercise, measured in millimeters.

### Values

| Category | Oldpeak Value |
|----------|---------------|
| **Normal** | 0 |
| **Mild Depression** | 0.1-1.0 mm |
| **Significant** | > 1.0 mm |
| **Severe** | > 2.0 mm |

### Clinical Significance

- ST depression indicates possible ischemia (reduced blood flow to heart)
- Higher values suggest more severe coronary artery disease
- **Target: 0 mm (no depression)**

---

## Model Evaluation Metrics

### AUC-ROC (Area Under the Curve)

| Range | Interpretation |
|-------|----------------|
| 0.9-1.0 | Excellent |
| 0.8-0.9 | Good |
| 0.7-0.8 | Fair |
| 0.6-0.7 | Poor |
| 0.5-0.6 | Fail (no discrimination) |

**Our Model: 0.798** (Good)

### ECE (Expected Calibration Error)

Measures how well predicted probabilities match actual outcomes.

| Range | Interpretation |
|-------|----------------|
| < 0.02 | Excellent calibration |
| 0.02-0.05 | Good calibration |
| 0.05-0.10 | Moderate calibration |
| > 0.10 | Poor calibration |

**Our Model: 0.019** (Excellent)

### Brier Score

Overall accuracy of probabilistic predictions (lower is better).

| Range | Interpretation |
|-------|----------------|
| < 0.1 | Excellent |
| 0.1-0.2 | Good |
| 0.2-0.3 | Fair |
| > 0.3 | Poor |

### Uncertainty (Standard Deviation)

- **Low (< 10%)**: Model is confident
- **Medium (10-20%)**: Model has some uncertainty
- **High (> 20%)**: Model recommends human verification

---

## Summary Table: Optimal Cardiovascular Health

| Metric | Optimal Value | Units |
|--------|---------------|-------|
| Blood Pressure | 120/80 | mmHg |
| Total Cholesterol | < 200 | mg/dL |
| LDL Cholesterol | < 100 | mg/dL |
| HDL Cholesterol | > 60 | mg/dL |
| Fasting Blood Sugar | < 100 | mg/dL |
| Resting Heart Rate | 60-70 | BPM |
| BMI | 18.5-24.9 | kg/m² |
| ST Depression | 0 | mm |

---

*This guide is for educational purposes. Always consult healthcare professionals for medical advice.*
