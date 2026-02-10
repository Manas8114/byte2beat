# 🫀 DeepMind-Inspired Cardiac Risk Assessment

## 🚀 Quick Start (One-Click Run)

### Windows

Double-click **`run.cmd`** in this folder.  
*(This trains the models if needed and launches the dashboard automatically.)*

### Mac/Linux

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Run Experiment (Train Models)
python run_experiment.py

# 3. Launch Dashboard
streamlit run dashboard.py
```

---

## 💡 Key Features for Judges

### 1. Uncertainty Quantification (Safety) 🛡️

Unlike standard ML models that blindly guess, our **Monte Carlo Dropout** network provides a **Confidence Interval**.

- **Try it:** Run `python tests/test_trust.py`
- **What it does:** Feeds "Garbage Data" (random noise) to the model.
- **Result:** The model correctly flags this as **High Uncertainty (>10%)**.

### 2. Foundation Model Approach 🧠

We use **TabPFN** (Prior-Data Fitted Network), a Transformer pre-trained on millions of datasets, to achieve "Zero-Shot" excellence on this small medical dataset.

### 3. "Clinician's Trust Cockpit" 🏥

The dashboard (`dashboard.py`) is designed for real-world usability:

- **"What-If" Analysis:** adjust sliders to see how weight loss or statins affect risk in real-time.
- **Glassmorphism UI:** A modern, distraction-free "DeepMind" aesthetic.
- **Narrative Generation:** Translates probabilities into plain English for doctors.

---

## 📊 Dataset & Ethics

### Data Source

This project uses the **UCI Heart Disease Dataset** (Cleveland subset) combined with cardiac failure data:

- **Primary:** `Data/Heart Attack/heart_processed.csv` (~900 patients)
- **Secondary:** `Data/Cardiac Failure/cardio_base.csv` (supplementary features)

### Ethical Considerations

> [!IMPORTANT]
> This is a **research prototype** for educational purposes only. It is NOT intended for clinical diagnosis.

- **Data Privacy:** All datasets are publicly available and anonymized
- **Bias Awareness:** The UCI dataset has known demographic imbalances (age/sex); predictions should be interpreted within this context
- **No Clinical Deployment:** This tool is designed for exploration and education, not patient care

---

## 📂 Project Structure

- `src/` - Core logic for Data Processing & Narratives.
- `models/` - Saved Joblib models (XGBoost, TabPFN, Uncertainty MLP).
- `tests/` - Automated tests including `test_trust.py` for safety validation.
- `dashboard.py` - Streamlit Frontend.
- `run_experiment.py` - Training pipeline with rigorous evaluation.

---

## ✅ Validation Results

Run `python tests/test_trust.py` to verify the safety mechanism:

```text
🧪 Starting Trust Test (Safety Check)...
✅ Uncertainty Model Loaded.
✅ Generated 20 rows of random noise data.

📊 Diagnostics:
   Risk Score:       ~50% (varies)
   Uncertainty (σ):  >10% (Target: >10%)

✅ PASS: High Uncertainty Detected
   Safety Mechanism: ACTIVE.
```

---

## 📋 Requirements

All dependencies are pinned for reproducibility. See `requirements.txt`.

**Core Stack:** Python 3.10+, PyTorch, XGBoost, TabPFN, Streamlit, SHAP
