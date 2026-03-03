# Project Plan: UncertaintyML Framework Understanding

## 1. Context & Objective

- **Goal:** Comprehensively analyze the architecture, capabilities, and clinical utility of the UncertaintyML system.
- **Current State:** The project is a robust medical risk assessment framework relying on TabPFN, MC Dropout, and FastAPI/Streamlit.

## 2. Component Breakdown

1. **Core Modeling Pipeline (`uncertaintyml/pipeline.py`, `models.py`)**
   - Analyze TabPFN foundation integration and MC Dropout Uncertainty scaling.
2. **Data & Adaptation (`uncertaintyml/data.py`, `adapters/`)**
   - Review how clinical concepts are mapped uniquely for diseases (e.g., Heart Disease vs. Diabetes).
3. **Interpretability Engine (`uncertaintyml/interpret.py`)**
   - Examine how SHAP values are bottlenecked into conceptual narratives.
4. **Serving Infrastructure (`api/server.py`, `dashboard.py`)**
   - Validate REST endpoints and Streamlit dashboard interactions.

## 3. Recommended Focus Areas for Enhancement

- **Performance Profiling:** ECE (Expected Calibration Error) and response time for N stochastic forward passes.
- **Security Audit:** FastAPI endpoint validation and potential PII leakage.
- **Test Coverage:** Unit and integration testing on the stochastic nature of predictions.

## 4. Required Agents for Full Analysis (Orchestration Phase 1)

- `explorer-agent`: Deep dive into core algorithmic files (`models.py`, `interpret.py`).
- `backend-specialist`: Analyze FastAPI router efficiency and Pydantic schemas.
- `security-auditor`: Verify data handling locally and test for PII boundaries.
- `performance-optimizer`: Evaluate the overhead of running 100 MC Dropout passes.
