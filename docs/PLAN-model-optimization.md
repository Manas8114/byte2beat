# PLAN: Model Optimization & Overfitting Prevention

**File:** `docs/PLAN-model-optimization.md`  
**Goal:** Detect and fix overfitting, tune hyperparameters, and improve model accuracy through principled regularization and validation.

---

## Context

### Current State

| Component | Status | Issue |
|---|---|---|
| `UncertaintyModel` (MC Dropout) | Trains for a **fixed** number of epochs | No training early stopping → potential overfit |
| `XGBClassifier` | Default hyperparameters | Untuned: likely suboptimal `max_depth`, `n_estimators`, `subsample` |
| `cross_validate_model()` | Exists in `evaluation.py` | **Never called** in `run_experiment.py` — overfitting never checked |
| Train/Val split | Simple random split | No cross-validation during training |
| Data | Imbalanced stroke dataset | Class imbalance not addressed (SMOTE / `scale_pos_weight`) |

### Key Findings

1. **MC Dropout training loop** (`models.py` L129-135) has no validation loss monitoring — it trains for all `epochs` regardless.
2. **XGBoost** is created with bare `XGBClassifier(eval_metric='logloss', random_state=42)` — no regularization (`reg_alpha`, `reg_lambda`, `min_child_weight`, `gamma`).
3. The `cross_validate_model()` utility is built, but `run_experiment.py` only does a single train/test split with no cross-validation.
4. The Stroke dataset is synthetically generated with class imbalance that's not handled.

---

## Phase 1: Overfitting Diagnosis Script

Create `scripts/check_overfitting.py`:

- Load trained models from `models/` and `models_stroke/`
- Compare **train AUC vs test AUC** (gap > 0.05 = likely overfit)
- Run **5-fold stratified cross-validation** for XGBoost and Uncertainty Model
- Report fold-level AUC mean ± std
- Plot learning curves (loss vs epoch) for Uncertainty Model

---

## Phase 2: XGBoost Regularization & Tuning

Modify `uncertaintyml/models.py` `ModelRegistry._register_defaults()`:

- Add regularized defaults: `max_depth=4`, `n_estimators=200`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`, `min_child_weight=3`
- Enable `early_stopping_rounds=15` via `eval_set` during fit (requires `PipelineConfig` change to pass a validation set)

---

## Phase 3: MC Dropout Training Early Stopping

Modify `UncertaintyModel.fit()` in `uncertaintyml/models.py`:

- Split training data internally (10% validation)  
- Track validation BCE loss per epoch
- Stop if validation loss doesn't improve for N epochs (patience=15)
- Restore best weights from the epoch with minimum val loss

---

## Phase 4: Class Imbalance Handling

Modify `uncertaintyml/adapters/stroke.py` and pipeline:

- Add `class_weight='balanced'` or compute `scale_pos_weight` = negative/positive count
- Pass to XGBoost via `scale_pos_weight` parameter
- For the MC Dropout network, use weighted `BCELoss`

---

## Phase 5: Feature Engineering

Add a preprocessing step in `DatasetAdapter.load()`:

- For Heart Disease: create interaction feature `Age × Cholesterol`, flag `BPxHR`
- For Stroke: encode `smoking_status` ordinally (never < formerly < smokes), create `risk_score_composite = hypertension + heart_disease + (age > 65)`

---

## Phase 6: Integration into `run_experiment.py`

- After training, run `cross_validate_model()` and print fold results
- Print train AUC vs test AUC gap
- Auto-flag if gap > 0.05 with `⚠️ POTENTIAL OVERFITTING`

---

## Verification Checklist

- [ ] `scripts/check_overfitting.py` outputs train/test AUC gap for all models
- [ ] XGBoost and MC Dropout both have tighter train/test gap after changes
- [ ] 5-fold CV AUC std < 0.05 (low variance = stable model)
- [ ] Class imbalance addressed: precision/recall for minority class (stroke=1) improves
- [ ] `pytest tests/` passes after all changes

---

## Expected Improvements

| Metric | Before | Expected After |
|---|---|---|
| XGBoost Train-Test AUC Gap | Unknown | < 0.05 |
| 5-Fold CV AUC | ~0.83 | ~0.86+ |
| Stroke minority class F1 | Low (imbalanced) | +10-15% |
| MC Dropout validation loss | Monotone decline → overfit | Early stop at optimal epoch |
