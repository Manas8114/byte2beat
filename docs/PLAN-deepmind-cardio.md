# PLAN-deepmind-cardio.md

> **Strategic Focus:** Leveraging Foundation Models, Uncertainty Quantification, and Concept-based Interpretability to build a "DeepMind-style" medical AI for Hack4Health.

## 0. Context & Application

* **Goal:** Build a hackathon-winning cardiovascular risk model that emphasizes **trustworthiness** and **representation learning** over simple classification accuracy.
* **Novelty Pitch:** "Motivated by recent Gaussian Process and Transformer work at DeepMind (2023-2025), we implement a hybrid architecture using TabPFN and Bayesian Uncertainty Estimation."
* **Constraint:** Must be reproducible on standard hardware (Colab/Laptop).

---

## 1. Architectural Pillars

### 🔹 Pillar A: Foundation Models for Tabular Data (Representation)

* **Technology:** **TabPFN** (Prior-Data Fitted Network).
* **Why:** Traditional ML (XGBoost) requires heavy tuning. TabPFN uses a Transformer pre-trained on millions of datasets to generate priors, enabling "Zero-Shot" or "Few-Shot" excellence on small medical datasets (~1k rows).
* **Implementation:** `tabpfn.TabPFNClassifier`.

### 🔹 Pillar B: Uncertainty Quantification (Trust)

* **Technology:** **Monte Carlo (MC) Dropout** / Bayesian Neural Networks.
* **Why:** Medical models must know *when they don't know*. A prediction of "High Risk" is useless if the confidence is 51%.
* **Implementation:** Custom PyTorch MLP with `dropout` active at inference time. Calculate $\mu$ (prediction) and $\sigma$ (uncertainty) over $N$ forward passes.

### 🔹 Pillar C: Concept-Bottleneck Interpretability (Why)

* **Technology:** **SHAP** grouped by **Clinical Concepts**.
* **Why:** Doctors don't care about "Feature 14". They care about "Lifestyle vs. Genetics".
* **Implementation:**
  * *Group 1 (Vitals):* BP, Cholesterol, MaxHR.
  * *Group 2 (Lifestyle):* Smoking, Alcohol, Activity, BMI.
  * *Group 3 (Demographics):* Age, Gender.
  * *Action:* Aggregate SHAP values to explain risk at the *Group* level.

---

## 2. Implementation Roadmap

### Phase 1: Data Infrastructure & Concept Mapping

- [ ] **Standardize Schema:** Merge/Clean `cardio_base.csv` and `heart_processed.csv`.
* [ ] **Feature Engineering:** Create explicit "Concept" dictionaries.
* [ ] **Pipeline:** Ensure robust `load_data()` that handles missing values via imputation (critical for small calibration sets).

### Phase 2: The "Hybrid" Model Suite

- [ ] **Baseline:** Implement XGBoost with rigorous cross-validation (5-fold).
* [ ] **Novelty:** Implement TabPFN. **Crucial:** Ensure it's running on CPU/GPU correctly (it's small enough for CPU inference).
* [ ] **Uncertainty:** Train the MC-Dropout MLP. Visualize "Out of Distribution" behavior (e.g., test on fake patient with age=120).

### Phase 3: Evaluation & Narrative

- [ ] **Metric 1:** AUC-ROC (Discriminative Power).
* [ ] **Metric 2:** **Calibration Curve** (Reliability). *DeepMind Focus.*
* [ ] **Metric 3:** **Prediction Interval Width** (Uncertainty quality).
* [ ] **Artifact:** Interpretation Grid – Sample patient reports showing "Risk: 85% ± 4% (High Confidence)" vs "Risk: 60% ± 25% (Low Confidence)".

---

## 3. Judge Perception Strategy

| Angle | How we win |
| :--- | :--- |
| **"Novelty"** | We aren't just tuning parameters; we are using **Priors** (TabPFN) and **Bayes** (Uncertainty). |
| **"Clinical Utility"** | We output **Confidence Intervals**, not just classes. This prevents false assurance. |
| **"Interpretability"** | We speak the doctor's language (Concepts) via our aggregated SHAP plots. |

## 4. Verification & Handoff

- [ ] **Notebook Check:** Ensure `cardio_hackathon_main.ipynb` runs top-to-bottom without error.
* [ ] **Report Check:** Verify the "Methodology" section uses the specific framing: *"Motivated by recent advances in uncertainty-aware medical AI..."*
