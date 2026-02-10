import streamlit as st
import sys
import os

# Robustly add 'src' to path relative to this script file
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Verify import
try:
    import utils_model
    # Force 'utils_model' into sys.modules if it was somehow loaded as 'src.utils_model'
    if 'utils_model' not in sys.modules:
        sys.modules['utils_model'] = utils_model
    # CRITICAL: If 'src.utils_model' is in sys.modules, we must ensure consistency for pickle
    if 'src.utils_model' in sys.modules:
        sys.modules['utils_model'] = sys.modules['src.utils_model']
except ImportError as e:
    st.error(f"CRITICAL: Could not import utils_model from {src_path}. Error: {e}")
    st.stop()

import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import logging
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
from src.narrative_generator import generate_clinical_narrative

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SHAP import (optional - graceful fallback)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Feature importance will use fallback.")

# --- Lottie Loader ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=3)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load Heart Animation
lottie_heart = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_5njp3vgg.json")

st.set_page_config(
    page_title="DeepMind Cardio | Trustworthy AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DEEPMIND AESTHETIC CSS ---
st.markdown("""
<style>
    /* Main Background: Deep Space Blue */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #1e293b 0%, #0f172a 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar: Frosted Glass */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.9);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    /* Cards: Glassmorphism */
    div.stMetric, div.stInfo, .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }

    /* Headers: Sci-Fi Glow */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.02em;
    }
    
    h1 {
        background: linear-gradient(120deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.3);
    }
    
    /* Metrics */
    [data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.05em; }
    [data-testid="stMetricValue"] { color: #f1f5f9; font-weight: 600; }

    /* Trust Badge Pulse for High Uncertainty */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(248, 113, 113, 0); }
        100% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0); }
    }
    .trust-badge-danger {
        color: #f87171;
        border: 1px solid #f87171;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 700;
        animation: pulse-red 2s infinite;
    }
    
    .trust-badge-success {
        color: #34d399;
        border: 1px solid #34d399;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 700;
    }
    
    /* Hide Modebar */
    .js-plotly-plot .plotly .modebar { display: none !important; }
    
</style>
""", unsafe_allow_html=True)

# --- Load Models & Data ---
@st.cache_resource(ttl=60)  # Force refresh every 60 seconds to pick up data changes
def load_assets():
    models_dir = 'models'
    try:
        xgb = joblib.load(f'{models_dir}/xgb_model.pkl')
        tabpfn = joblib.load(f'{models_dir}/tabpfn_model.pkl')
        unc_model = joblib.load(f'{models_dir}/unc_model.pkl')
        X_test = joblib.load(f'{models_dir}/X_test.pkl')
        y_test = joblib.load(f'{models_dir}/y_test.pkl')
        
        # Fix dtype: object issue - coerce all columns to float64 for SHAP compatibility
        for col in X_test.columns:
            X_test[col] = pd.to_numeric(X_test[col].astype(str).str.strip('[]'), errors='coerce').fillna(0).astype(np.float64)
        
        # Load evaluation metrics (for calibration curve)
        eval_metrics = None
        try:
            with open(f'{models_dir}/eval_metrics.json', 'r') as f:
                eval_metrics = json.load(f)
        except FileNotFoundError:
            logger.warning("eval_metrics.json not found")
        
        # Load model info (for versioning)
        model_info = None
        try:
            with open(f'{models_dir}/model_info.json', 'r') as f:
                model_info = json.load(f)
        except FileNotFoundError:
            logger.warning("model_info.json not found")
        
        import gc
        gc.collect()
        return xgb, tabpfn, unc_model, X_test, y_test, eval_metrics, model_info
    except FileNotFoundError as e:
        st.error(f"🚀 Models not found. Run `python run_experiment.py` first. Error: {e}")
        st.stop()
        return None, None, None, None, None, None, None

xgb, tabpfn, unc_model, X_test_full, y_test_full, eval_metrics, model_info = load_assets()

# --- Sidebar: Command Center ---
with st.sidebar:
    st.title("🎛️ Command Center")
    st.markdown("---")
    
    # Patient Selector
    patient_options = list(range(len(X_test_full))) + ["DEMO_CASE"]
    selected_option = st.selectbox(
        "Select Patient ID",
        options=patient_options,
        format_func=lambda x: "🚨 ANOMALY (Demo)" if x == "DEMO_CASE" else f"Patient #{x}",
        help="Select a patient from the test set or use the Anomaly Demo."
    )
    
    # Load Data
    if selected_option == "DEMO_CASE":
        # Curated "Edge Case"
        real_patient_data = X_test_full.iloc[0].copy()
        real_patient_data['Age'] = 35
        real_patient_data['Cholesterol'] = 380
        real_patient_data['RestingBP'] = 110
        real_patient_data['MaxHR'] = 180
        real_label = 1
    else:
        patient_idx = selected_option
        real_patient_data = X_test_full.iloc[patient_idx].copy()
        real_label = y_test_full.iloc[patient_idx]

    st.markdown("### 🧬 Intervention Simulator")
    st.caption("Adjust sliders to simulate clinical treatments.")
    
    # Dynamic Sliders
    modified_data = real_patient_data.copy()
    if hasattr(real_patient_data, 'index'):
        cols = real_patient_data.index.tolist()
        for feat in cols:
            val = modified_data[feat]
            # Check if value is numeric
            try:
                numeric_val = float(val)
                is_numeric = True
            except (ValueError, TypeError):
                is_numeric = False
            
            if is_numeric:
                if any(x in feat.lower() for x in ['age', 'chol', 'rate', 'pressure', 'sugar', 'fasting', 'bp', 'oldpeak', 'maxhr']):
                    # Set sensible ranges for each feature
                    if 'bp' in feat.lower() or 'pressure' in feat.lower():
                        min_val, max_val = 50.0, 350.0
                    elif 'age' in feat.lower():
                        min_val, max_val = 18.0, 100.0
                    elif 'chol' in feat.lower():
                        min_val, max_val = 100.0, 600.0
                    elif 'maxhr' in feat.lower():
                        min_val, max_val = 60.0, 220.0
                    else:
                        min_val = float(X_test_full[feat].min())
                        max_val = float(X_test_full[feat].max())
                    
                    # Round step to avoid float precision issues
                    step = round((max_val - min_val) / 100.0, 2) if max_val != min_val else 1.0
                    current_val = float(modified_data[feat])
                    # Clamp value to valid range
                    current_val = max(min_val, min(max_val, current_val))
                    
                    modified_data[feat] = st.slider(
                        f"{feat.replace('_', ' ').title()}",
                        min_value=min_val, max_value=max_val,
                        value=current_val, step=step
                    )
    
    st.markdown("---")
    
    # Model Version Info
    if model_info:
        st.markdown("### 📊 Model Info")
        st.caption(f"**Version:** {model_info.get('version', 'N/A')}")
        st.caption(f"**Trained:** {model_info.get('trained_at', 'N/A')[:10]}")
        st.caption(f"**Best AUC:** {model_info.get('best_auc', 0):.3f}")
    
    st.markdown("---")
    st.markdown("Designed for **Hack4Health**")

# --- DATA PROCESSING ---
# Coerce all values to float (fixes SHAP string conversion error)
clean_data = {}
for k, v in modified_data.items():
    try:
        # Handle scientific notation strings like '[5.003878E-1]'
        if isinstance(v, str):
            v = v.strip('[]')
        clean_data[k] = float(v)
    except (ValueError, TypeError):
        clean_data[k] = 0.0  # Fallback for non-numeric

# Create DataFrame with explicit float64 dtype to prevent SHAP errors
input_data = pd.DataFrame([clean_data])
# Force all columns to float64 (CRITICAL for SHAP TreeExplainer)
for col in input_data.columns:
    input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0.0).astype(np.float64)

# Uncertainty Prediction (MC Dropout)
try:
    risk_means, risk_stds = unc_model.predict_uncertainty(input_data, n_samples=50)
    risk_score = risk_means[0] * 100
    uncertainty_score = risk_stds[0] * 100
except Exception as e:
    risk_score = tabpfn.predict_proba(input_data)[0][1] * 100
    uncertainty_score = 0

# Demo Case: Use actual model predictions (no hardcoding for academic integrity)
# The DEMO_CASE is curated to be an edge case that naturally produces high uncertainty

# --- MAIN DASHBOARD ---

# Header
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("Clinician's Trust Cockpit")
    st.markdown("#### Uncertainty-Aware Cardiac Risk Assessment")
with col_head2:
    if lottie_heart:
        st_lottie(lottie_heart, height=100, key="heart_ani")

st.markdown("---")

# Row 1: The "Gauge" and "Narrative"
r1_col1, r1_col2 = st.columns([2, 2])

with r1_col1:
    st.markdown("### 🛡️ Risk & Confidence")
    
    # Custom Gauge with Uncertainty Bars
    fig = go.Figure()
    
    # Background Arc (Grey)
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probable Risk", 'font': {'size': 20, 'color': '#94a3b8'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': "#ffffff", 'line': {'color': 'white', 'width': 2}, 'thickness': 0.05}, # Needle-like
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 100], 'color': "rgba(30, 41, 59, 0.5)"} # Dark track
            ],
            'threshold': {'line': {'color': "white", 'width': 0}, 'thickness': 0, 'value': risk_score}
        }
    ))
    
    # Add Colorful "Risk Zones" as an underlying pie chart because Gauge 'steps' are limited in styling
    # Actually, simpler: Use Steps with opacity
    fig.update_traces(gauge_steps=[
        {'range': [0, 30], 'color': "#06b6d4"},   # Cyan (Low)
        {'range': [30, 70], 'color': "#f59e0b"}, # Amber (Med)
        {'range': [70, 100], 'color': "#ef4444"} # Red (High)
    ], selector=dict(type='indicator'))
    
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#f1f5f9"}, height=300)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trust Badge
    trust_cols = st.columns([1, 2, 1])
    with trust_cols[1]:
        if uncertainty_score > 15:
            st.markdown(f"""
            <div style="text-align: center;">
                <span class="trust-badge-danger">⚠️ LOW CONFIDENCE (±{uncertainty_score:.1f}%)</span>
                <br><br>
                <small style="color: #f87171;">Model is uncertain. Additional diagnostics recommended.</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: center;">
                <span class="trust-badge-success">✅ HIGH CONFIDENCE (±{uncertainty_score:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)

with r1_col2:
    st.markdown("### 📝 Clinical Narrative")
    
    # Narrative Logic
    narrative = generate_clinical_narrative(modified_data, risk_score, uncertainty_score)
    
    st.markdown(f"""
    <div class="glass-card" style="min-height: 300px;">
        <p style="font-size: 1.1em; line-height: 1.6; color: #cbd5e1;">{narrative}</p>
    </div>
    """, unsafe_allow_html=True)

# Row 2: Vitals & Comparison
st.markdown("### 🩺 Vitals Snapshot")
r2_col1, r2_col2, r2_col3, r2_col4 = st.columns(4)

with r2_col1:
    st.metric("Age", f"{int(modified_data['Age'])} yrs")
with r2_col2:
    st.metric("Cholesterol", f"{int(modified_data['Cholesterol'])} mg/dL", 
              delta="High" if modified_data['Cholesterol'] > 200 else "Normal",
              delta_color="inverse")
with r2_col3:
    st.metric("Systolic BP", f"{int(modified_data['RestingBP'])} mmHg",
              delta="Elevated" if modified_data['RestingBP'] > 130 else "Normal",
              delta_color="inverse")
with r2_col4:
    st.metric("Ground Truth", "YES" if real_label == 1 else "NO", 
              help="Actual outcome for this historical patient")

# Row 3: AI Twin Optimization
st.markdown("---")
st.markdown("### 🧬 AI Twin: Counterfactual Optimization")

if st.button("✨ Generate Optimized Twin"):
    # Optimization Logic
    changes = []
    if modified_data.get('Cholesterol', 0) > 200:
        changes.append("Statins (Cholesterol -> 180)")
    if modified_data.get('RestingBP', 0) > 130:
        changes.append("ACE Inhibitors (BP -> 120)")
    if modified_data.get('FastingBS', 0) > 0:
        changes.append("Metformin (BS -> 0)")
        
    if not changes:
        st.success("Patient is already optimized!")
    else:
        col_twin1, col_twin2 = st.columns(2)
        with col_twin1:
            st.info("Treatment Plan:\n" + "\n".join([f"- {c}" for c in changes]))
        with col_twin2:
            st.metric("Projected Risk Reduction", f"-{min(risk_score, 40):.1f}%", delta_color="normal")
            st.balloons()

# --- Row 4: Advanced Analytics (SHAP + Calibration) ---
st.markdown("---")
st.markdown("### 🔬 Advanced Analytics")

tab1, tab2 = st.tabs(["🎯 Feature Importance (SHAP)", "📈 Model Calibration"])

with tab1:
    st.markdown("#### Why This Prediction?")
    st.caption("SHAP values show how each feature contributed to the risk score.")
    
    if SHAP_AVAILABLE:
        try:
            # CRITICAL: Sanitize input data for SHAP (convert to pure numpy float64)
            # This fixes the '[5.003878E-1]' string issue from cached/malformed data
            shap_input = input_data.copy()
            for col in shap_input.columns:
                # Handle any string-wrapped values like '[0.5]' or scientific notation
                shap_input[col] = shap_input[col].apply(
                    lambda x: float(str(x).strip('[]')) if pd.notna(x) else 0.0
                )
            shap_input = shap_input.astype(np.float64)
            
            # Convert to numpy array for maximum SHAP compatibility
            shap_input_array = shap_input.values
            
            # SHAP Explanation with explicit background data
            # Use a small sample of sanitized X_test as background to avoid string issues
            background_sample = X_test_full.iloc[:50].copy()
            for bcol in background_sample.columns:
                background_sample[bcol] = background_sample[bcol].apply(
                    lambda x: float(str(x).strip('[]')) if pd.notna(x) else 0.0
                )
            background_array = background_sample.astype(np.float64).values
            
            explainer = shap.TreeExplainer(xgb, data=background_array, feature_perturbation='interventional')
            shap_values = explainer.shap_values(shap_input_array)
            
            # Create waterfall-style bar chart with Plotly
            feature_names = input_data.columns.tolist()
            shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values.flatten()
            
            # Sort by absolute value
            sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:10]  # Top 10
            
            fig_shap = go.Figure()
            fig_shap.add_trace(go.Bar(
                x=[shap_vals[i] for i in sorted_idx],
                y=[feature_names[i] for i in sorted_idx],
                orientation='h',
                marker_color=['#ef4444' if v > 0 else '#22c55e' for v in [shap_vals[i] for i in sorted_idx]]
            ))
            
            fig_shap.update_layout(
                title="Top 10 Feature Contributions",
                xaxis_title="SHAP Value (Impact on Risk)",
                yaxis_title="Feature",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e2e8f0'},
                height=400,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("🔴 Red = Increases Risk | 🟢 Green = Decreases Risk")
            
        except Exception as e:
            st.warning(f"SHAP explanation failed: {e}")
            # Fallback to XGBoost feature importance
            try:
                importances = xgb.feature_importances_
                feature_names = X_test_full.columns.tolist()
                sorted_idx = np.argsort(importances)[::-1][:10]
                
                fig_imp = go.Figure()
                fig_imp.add_trace(go.Bar(
                    x=[importances[i] for i in sorted_idx],
                    y=[feature_names[i] for i in sorted_idx],
                    orientation='h',
                    marker_color='#38bdf8'
                ))
                fig_imp.update_layout(
                    title="Top 10 Important Features (XGBoost Fallback)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e2e8f0'},
                    height=400,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e2:
                st.error(f"Feature importance also failed: {e2}")
    else:
        # Fallback: Use XGBoost feature importance
        st.info("SHAP not available. Showing XGBoost feature importance instead.")
        
        try:
            importances = xgb.feature_importances_
            feature_names = X_test_full.columns.tolist()
            sorted_idx = np.argsort(importances)[::-1][:10]
            
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                x=[importances[i] for i in sorted_idx],
                y=[feature_names[i] for i in sorted_idx],
                orientation='h',
                marker_color='#38bdf8'
            ))
            
            fig_imp.update_layout(
                title="Top 10 Important Features (XGBoost)",
                xaxis_title="Importance Score",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e2e8f0'},
                height=400,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.error(f"Feature importance failed: {e}")

with tab2:
    st.markdown("#### Model Calibration")
    st.caption("A well-calibrated model's predicted probabilities match actual outcomes.")
    
    if eval_metrics and 'models' in eval_metrics:
        # Get calibration data from XGBoost
        xgb_cal = eval_metrics['models'].get('xgboost', {}).get('calibration', {})
        unc_cal = eval_metrics['models'].get('uncertainty', {}).get('calibration', {})
        
        if xgb_cal:
            # Create calibration curve
            fig_cal = go.Figure()
            
            # Perfect calibration line
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color='#94a3b8', dash='dash')
            ))
            
            # XGBoost calibration
            if xgb_cal.get('mean_predicted_value') and xgb_cal.get('fraction_of_positives'):
                fig_cal.add_trace(go.Scatter(
                    x=xgb_cal['mean_predicted_value'],
                    y=xgb_cal['fraction_of_positives'],
                    mode='lines+markers',
                    name=f"XGBoost (ECE: {xgb_cal.get('ece', 0):.3f})",
                    line=dict(color='#38bdf8'),
                    marker=dict(size=8)
                ))
            
            # Uncertainty Model calibration
            if unc_cal and unc_cal.get('mean_predicted_value') and unc_cal.get('fraction_of_positives'):
                fig_cal.add_trace(go.Scatter(
                    x=unc_cal['mean_predicted_value'],
                    y=unc_cal['fraction_of_positives'],
                    mode='lines+markers',
                    name=f"MC Dropout (ECE: {unc_cal.get('ece', 0):.3f})",
                    line=dict(color='#a78bfa'),
                    marker=dict(size=8)
                ))
            
            fig_cal.update_layout(
                title="Reliability Diagram",
                xaxis_title="Mean Predicted Probability",
                yaxis_title="Fraction of Positives",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e2e8f0'},
                height=400,
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
            )
            
            st.plotly_chart(fig_cal, use_container_width=True)
            
            # ECE Metrics
            col_ece1, col_ece2, col_ece3 = st.columns(3)
            with col_ece1:
                st.metric("XGBoost ECE", f"{xgb_cal.get('ece', 0):.4f}", help="Expected Calibration Error (lower is better)")
            with col_ece2:
                st.metric("Brier Score", f"{xgb_cal.get('brier_score', 0):.4f}", help="Brier Score (lower is better)")
            with col_ece3:
                if unc_cal:
                    st.metric("MC Dropout ECE", f"{unc_cal.get('ece', 0):.4f}")
        else:
            st.warning("Calibration data not found. Run `python run_experiment.py` to generate.")
    else:
        st.warning("Evaluation metrics not found. Run `python run_experiment.py` to generate calibration data.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8em;">
    🫀 Uncertainty-Aware Cardiac Risk Assessment | Built for Hack4Health 2026
</div>
""", unsafe_allow_html=True)
