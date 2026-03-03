# Project Plan: Fix 100% Risk Bug (Engineered Features Contradiction)

## 🧠 Brainstorm: Solving the 100% Risk Bug

### Context

When the user lowers the sliders in the dashboard (e.g., `Age` to 18, `RestingBP` to 120), the model still predicts 100% risk. This is because **engineered features** (like `Age_x_Cholesterol`, `BP_HR_ratio`, `Hypertension`, and `Diabetes`) are NOT being updated when the base sliders are moved. The model receives a contradictory profile (e.g., a "healthy" 18-year-old with the `Age_x_Cholesterol` score of an unhealthy 80-year-old), which breaks the model's logic and causes extreme uncertainty and 100% risk predictions.

---

### Option A: Dynamic Feature Recalculation in Dashboard

Recalculate all engineered features (`Age_x_Cholesterol`, `BP_HR_ratio`, `Hypertension`, `Diabetes`) dynamically inside `dashboard.py` every time a slider is moved, before passing the data to the model.

✅ **Pros:**

- Keeps the models highly accurate (they still get to use the predictive power of engineered features).
- The dashboard reflects true "What-If" scenarios accurately.

❌ **Cons:**

- Requires hardcoding the feature engineering logic from `utils_data.py` into `dashboard.py`.

📊 **Effort:** Low

---

### Option B: Drop Engineered Features & Retrain Models

Remove `Age_x_Cholesterol`, `BP_HR_ratio`, `Hypertension`, and `Diabetes` from the dataset entirely, and just let the model learn purely from the raw base features.

✅ **Pros:**

- Simplifies the architecture. `dashboard.py` won't need any complex logic.

❌ **Cons:**

- We lose the accuracy boost that comes from explicit medical indicators (like Diabetes).
- Requires a full ML pipeline retrain.

📊 **Effort:** Medium

---

### Option C: Expose Engineered Features as Manual Sliders

Instead of recalculating them behind the scenes, give the user manual control over `Age_x_Cholesterol` and `BP_HR_ratio` on the UI.

✅ **Pros:**

- No code change required to inference logic.

❌ **Cons:**

- Terrible UX. Doctors shouldn't have to manually calculate and input an "Age x Cholesterol" ratio.

📊 **Effort:** Low

---

## 💡 Recommendation

**Option A** because it maintains the high accuracy of our new features while providing a seamless, mathematically correct UX for the clinician.

---

## Task Breakdown (Option A Implementation)

1. **Update `dashboard.py`**
   - Add recalculation logic in the `DATA PROCESSING` section before `input_data` is sent to the models.
   - Example:`clean_data['Hypertension'] = 1 if clean_data.get('RestingBP', 0) > 130 else 0`
2. **Review Other Datasets**
   - Ensure the Stroke dataset's engineered features (if any) are also dynamically calculated if we switch the dropdown.
3. **Verify**
   - Confirm that moving the Age to 18 and BP to 120 now returns a low risk score (< 5%).

## Verification Checklist

- [ ] Engineered features accurately mirror base features on every slider change.
- [ ] Risk drops to logical baselines when "Reset to Normal Ranges" is clicked.
- [ ] SHAP values calculate without error.

---
**Agent Assignment:** `frontend-specialist` or `backend-specialist` for the dashboard Python changes.
