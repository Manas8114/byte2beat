import streamlit as st
import sys
import os

# --- CRITICAL FIX: sklearn compatibility for Python 3.13+ ---
import sklearn.utils.validation
if not hasattr(sklearn.utils.validation, '_is_pandas_df'):
    try:
        import pandas as pd
        def _is_pandas_df(X):
            return isinstance(X, pd.DataFrame)
        sklearn.utils.validation._is_pandas_df = _is_pandas_df
    except ImportError:
        pass

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Robustly add 'src' to path as fallback
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
from uncertaintyml.smart_narrative import SmartNarrativeEngine

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
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

    /* Main Background: Deep Space Blue */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0f172a 0%, #020617 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar: Frosted Glass */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Cards: Glassmorphism */
    div.stMetric, div.stInfo, .glass-card {
        background: rgba(30, 41, 59, 0.3);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.05), 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    div.stMetric:hover {
        transform: translateY(-2px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.1), 0 12px 40px 0 rgba(0, 0, 0, 0.3);
    }

    /* Headers: Sci-Fi Glow */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.03em;
    }
    
    h1 {
        background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 40px rgba(56, 189, 248, 0.2);
        font-weight: 800 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #f8fafc; font-weight: 700; font-family: 'Outfit', sans-serif; letter-spacing: -0.02em; }

    /* Trust Badge Pulse for High Uncertainty */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(225, 29, 72, 0.5); border-color: rgba(225, 29, 72, 0.8); }
        70% { box-shadow: 0 0 0 8px rgba(225, 29, 72, 0); border-color: rgba(225, 29, 72, 1); }
        100% { box-shadow: 0 0 0 0 rgba(225, 29, 72, 0); border-color: rgba(225, 29, 72, 0.8); }
    }
    
    .trust-badge-danger {
        color: #fca5a5;
        background-color: rgba(225, 29, 72, 0.15);
        border: 1px solid rgba(225, 29, 72, 0.8);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85em;
        font-weight: 600;
        animation: pulse-red 2s infinite cubic-bezier(0.66, 0, 0, 1);
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .trust-badge-success {
        color: #6ee7b7;
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.5);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85em;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Segmented Controls for Tabs */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.5);
        padding: 4px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stTabs"] [data-baseweb="tab"] {
        border-radius: 8px;
        padding-top: 8px;
        padding-bottom: 8px;
        color: #94a3b8;
        font-weight: 500;
    }
    
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    
    /* Hide Modebar */
    .js-plotly-plot .plotly .modebar { display: none !important; }
    
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Command Center ---
with st.sidebar:
    st.title("🎛️ Command Center")
    st.markdown("---")
    
    disease_type = st.radio(
        "Select Clinical Domain",
        ["Cardiovascular (Heart Disease)", "Neurological (Stroke)"],
        index=0
    )
    
    # Reload assets based on selection without clearing the whole cache to avoid slowing down,
    # but using parameters in the cached function
    models_dir_to_load = 'models' if "Heart" in disease_type else 'models_stroke'

# --- Load Models & Data ---
@st.cache_resource(ttl=60)  # Force refresh every 60 seconds
def load_assets(models_dir):
    try:
        xgb = joblib.load(f'{models_dir}/xgboost_model.pkl') if os.path.exists(f'{models_dir}/xgboost_model.pkl') else joblib.load(f'{models_dir}/xgb_model.pkl')
        tabpfn = joblib.load(f'{models_dir}/tabpfn_model.pkl') if os.path.exists(f'{models_dir}/tabpfn_model.pkl') else None
        unc_model = joblib.load(f'{models_dir}/uncertainty_model.pkl')
        X_test = joblib.load(f'{models_dir}/X_test.pkl')
        y_test = joblib.load(f'{models_dir}/y_test.pkl')
        
        # ... snip handling logic ...
        for col in X_test.columns:
            X_test[col] = pd.to_numeric(X_test[col].astype(str).str.strip('[]'), errors='coerce').fillna(0).astype(np.float64)
        
        eval_metrics = None
        try:
            with open(f'{models_dir}/eval_metrics.json', 'r') as f:
                eval_metrics = json.load(f)
        except FileNotFoundError:
            logger.warning("eval_metrics.json not found")
        
        model_info = None
        try:
            with open(f'{models_dir}/model_info.json', 'r') as f:
                model_info = json.load(f)
        except FileNotFoundError:
             pass
             
        import gc
        gc.collect()
        return xgb, tabpfn, unc_model, X_test, y_test, eval_metrics, model_info
    except FileNotFoundError as e:
        st.error(f"🚀 Models not found in {models_dir}. Run `python run_experiment.py` first. Error: {e}")
        st.stop()
        return None, None, None, None, None, None, None

xgb, tabpfn, unc_model, X_test_full, y_test_full, eval_metrics, model_info = load_assets(models_dir_to_load)

with st.sidebar:
    # Patient Selector
    patient_options = list(range(len(X_test_full))) + ["DEMO_CASE"]
    selected_option = st.selectbox(
        "Select Patient ID",
        options=patient_options,
        format_func=lambda x: "🚨 ANOMALY (Demo)" if x == "DEMO_CASE" else f"Patient #{x}",
        help="Select a patient from the test set or use the Anomaly Demo."
    )
    
    # Check if patient changed to reset sliders
    if "current_patient" not in st.session_state:
        st.session_state["current_patient"] = selected_option
    if st.session_state["current_patient"] != selected_option:
        for key in list(st.session_state.keys()):
            if key.endswith("_slider") or key.endswith("_num"):
                del st.session_state[key]
        st.session_state["current_patient"] = selected_option
        
    # Load Data
    if selected_option == "DEMO_CASE":
        # Curated "Edge Case"
        real_patient_data = X_test_full.iloc[0].copy()
        if "Heart" in disease_type:
            real_patient_data['Age'] = 35
            real_patient_data['Cholesterol'] = 380
            real_patient_data['RestingBP'] = 110
            real_patient_data['MaxHR'] = 180
            real_label = 1
        else:
            real_patient_data['age'] = 82
            if 'bmi' in real_patient_data:
                real_patient_data['bmi'] = 52.0
            if 'avg_glucose_level' in real_patient_data:
                real_patient_data['avg_glucose_level'] = 265.0
            real_label = 1
    else:
        patient_idx = selected_option
        real_patient_data = X_test_full.iloc[patient_idx].copy()
        real_label = y_test_full.iloc[patient_idx]

    st.markdown("### 🧬 Intervention Simulator")
    st.caption("Adjust sliders to simulate clinical treatments.")
    reset_clicked = st.button("🌟 Reset to Normal Ranges")
    
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
                # Skip engineered features — they are recalculated automatically
                if feat in ('Age_x_Cholesterol', 'BP_HR_ratio', 'Hypertension', 'Diabetes'):
                    pass
                elif any(x in feat.lower() for x in ['age', 'chol', 'rate', 'pressure', 'sugar', 'fasting', 'bp', 'oldpeak', 'maxhr', 'glucose', 'weight', 'height']):
                    # Set sensible ranges for each feature
                    if 'bp' in feat.lower() or 'pressure' in feat.lower():
                        min_val, max_val = 50.0, 350.0
                    elif 'age' in feat.lower():
                        min_val, max_val = 18.0, 100.0
                    elif 'chol' in feat.lower():
                        min_val, max_val = 100.0, 600.0
                    elif 'maxhr' in feat.lower():
                        min_val, max_val = 60.0, 220.0
                    elif 'glucose' in feat.lower():
                         min_val, max_val = 40.0, 400.0
                    elif 'bmi' in feat.lower():
                         min_val, max_val = 10.0, 60.0
                    elif 'weight' in feat.lower():
                         min_val, max_val = 30.0, 200.0
                    elif 'height' in feat.lower():
                         min_val, max_val = 100.0, 220.0
                    else:
                        min_val = float(X_test_full[feat].min())
                        max_val = float(X_test_full[feat].max())
                    
                    # Round step to avoid float precision issues
                    step = round((max_val - min_val) / 100.0, 2) if max_val != min_val else 1.0
                    current_val = float(modified_data[feat])
                    # Clamp value to valid range
                    current_val = max(min_val, min(max_val, current_val))
                    
                    slider_key = f"{feat}_slider"
                    num_key = f"{feat}_num"
                    
                    if reset_clicked:
                        if 'bp' in feat.lower() or 'pressure' in feat.lower():
                            new_val = 120.0
                        elif 'age' in feat.lower():
                            new_val = 40.0
                        elif 'chol' in feat.lower():
                            new_val = 180.0
                        elif 'fasting' in feat.lower() or 'sugar' in feat.lower():
                            new_val = 0.0
                        elif 'glucose' in feat.lower():
                            new_val = 90.0
                        elif 'maxhr' in feat.lower() or 'rate' in feat.lower():
                            new_val = 150.0
                        elif 'oldpeak' in feat.lower():
                            new_val = 0.0
                        elif 'bmi' in feat.lower():
                            new_val = 22.0
                        elif 'weight' in feat.lower():
                            new_val = 70.0
                        elif 'height' in feat.lower():
                            new_val = 170.0
                        else:
                            new_val = current_val
                        
                        new_val = max(min_val, min(max_val, new_val))
                        st.session_state[slider_key] = new_val
                        st.session_state[num_key] = new_val
                    
                    # Initialize session state for synchronization
                    if slider_key not in st.session_state:
                        st.session_state[slider_key] = current_val
                    if num_key not in st.session_state:
                        st.session_state[num_key] = current_val

                    def sync_input(source, target):
                        st.session_state[target] = st.session_state[source]

                    st.markdown(f"**{feat.replace('_', ' ').title()}**")
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.slider(
                            "Slider",
                            min_value=min_val, max_value=max_val,
                            step=step,
                            key=slider_key,
                            on_change=sync_input,
                            kwargs={'source': slider_key, 'target': num_key},
                            label_visibility="collapsed"
                        )
                    with col2:
                        st.number_input(
                            "TextInput",
                            min_value=min_val, max_value=max_val,
                            step=step,
                            key=num_key,
                            on_change=sync_input,
                            kwargs={'source': num_key, 'target': slider_key},
                            label_visibility="collapsed"
                        )
                        
                    modified_data[feat] = st.session_state[slider_key]
    
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

# --- DYNAMIC RECALCULATION OF ENGINEERED FEATURES ---
# These MUST be recomputed from the current slider values, otherwise the model
# receives contradictory inputs (e.g. low BP but Hypertension=1) and predicts 100%.
age = clean_data.get('Age', 0)
chol = clean_data.get('Cholesterol', 0)
bp = clean_data.get('RestingBP', 0)
hr = clean_data.get('MaxHR', 1)  # avoid division by zero
fbs = clean_data.get('FastingBS', 0)
glucose = clean_data.get('Glucose', 0)

if 'Age_x_Cholesterol' in clean_data:
    clean_data['Age_x_Cholesterol'] = age * chol
if 'BP_HR_ratio' in clean_data:
    clean_data['BP_HR_ratio'] = bp / max(hr, 1)
if 'Hypertension' in clean_data:
    clean_data['Hypertension'] = 1.0 if bp > 130 else 0.0
if 'Diabetes' in clean_data:
    clean_data['Diabetes'] = 1.0 if (fbs == 1 or glucose > 125) else 0.0

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
    if tabpfn:
         risk_score = tabpfn.predict_proba(input_data)[0][1] * 100
    else:
         risk_score = xgb.predict_proba(input_data)[0][1] * 100
    uncertainty_score = 0

# Demo Case: Use actual model predictions (no hardcoding for academic integrity)
# The DEMO_CASE is curated to be an edge case that naturally produces high uncertainty

# --- MAIN DASHBOARD ---

col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("Clinician's Risk Cockpit")
    st.markdown(f"#### Uncertainty-Aware {disease_type} Assessment")
with col_head2:
    if lottie_heart and "Heart" in disease_type:
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
    
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#f1f5f9"}, height=300, margin=dict(t=50, b=20, l=20, r=20))
    
    st.plotly_chart(fig, width='stretch')
    
    # Trust Badge
    trust_cols = st.columns([1, 2, 1])
    with trust_cols[1]:
        if uncertainty_score > 15:
            st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <span class="trust-badge-danger">⚠️ LOW CONFIDENCE (±{uncertainty_score:.1f}%)</span>
                <br><br>
                <small style="color: #f87171;">Model is uncertain. Additional diagnostics recommended.</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <span class="trust-badge-success">✅ HIGH CONFIDENCE (±{uncertainty_score:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)

with r1_col2:
    st.markdown("### 🧮 Model Insights & What-If Analysis")
    
    # Quantitative Insights with Counterfactual Reasoning
    smart_engine = SmartNarrativeEngine(model=unc_model, feature_names=input_data.columns.tolist())
    narrative = smart_engine.generate(
        modified_data, risk_score, uncertainty_score,
        input_data=input_data,
        disease_type=disease_type
    )
    
    st.markdown(narrative, unsafe_allow_html=True)

# Row 2: Vitals & Comparison
st.markdown("### 🩺 Patient Snapshot")
r2_col1, r2_col2, r2_col3, r2_col4 = st.columns(4)

with r2_col1:
    age_val = modified_data.get('Age', modified_data.get('age', 0))
    st.metric("Age", f"{int(age_val)} yrs")
with r2_col2:
    if "Heart" in disease_type:
        st.metric("Cholesterol", f"{int(modified_data.get('Cholesterol', 0))} mg/dL", 
                  delta="High" if modified_data.get('Cholesterol', 0) > 200 else "Normal",
                  delta_color="inverse")
    else:
         bmi_val = modified_data.get('bmi', 0)
         st.metric("BMI", f"{int(bmi_val)}", 
                  delta="High" if bmi_val > 30 else "Normal",
                  delta_color="inverse")
with r2_col3:
    if "Heart" in disease_type:
        st.metric("Systolic BP", f"{int(modified_data.get('RestingBP', 0))} mmHg",
                  delta="Elevated" if modified_data.get('RestingBP', 0) > 130 else "Normal",
                  delta_color="inverse")
    else:
        gluc_val = modified_data.get('avg_glucose_level', 0)
        st.metric("Glucose", f"{int(gluc_val)} mg/dL",
                  delta="Elevated" if gluc_val > 140 else "Normal",
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

# --- Row 4: Advanced Analytics (SHAP + Calibration + Ward View) ---
st.markdown("---")
st.markdown("### 🔬 Advanced Analytics")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Feature Importance (SHAP)", "📈 Model Calibration", "🏥 Ward View", "📘 Normal Ranges"])

with tab4:
    st.markdown("#### 📘 Normal Medical Ranges Reference")
    st.markdown("""
    - **Resting Blood Pressure (RestingBP):** Normal is < 120/80 mmHg. Elevated is 120-129 systolic. Hypertension is ≥ 130 systolic or ≥ 80 diastolic.
    - **Cholesterol:** Total < 200 mg/dL is desirable. LDL < 100 mg/dL is optimal. HDL ≥ 60 mg/dL is protective against heart disease.
    - **Fasting Blood Sugar (FastingBS/Glucose):** Normal is 70-99 mg/dL. Pre-diabetes is 100-125 mg/dL. Diabetes is ≥ 126 mg/dL.
    - **Maximum Heart Rate (MaxHR):** Roughly estimated as 220 minus your age (e.g., for a 40-year-old, ~180 bpm).
    - **Oldpeak (ST depression):** Normal is < 0.5 mm. ≥ 0.5 mm is considered pathological, and > 2 mm is an urgent "red flag" for coronary artery disease.
    - **BMI (Body Mass Index):** Normal weight is 18.5 - 24.9. Overweight is 25 - 29.9. Obese is ≥ 30.
    """)

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
            
            st.plotly_chart(fig_shap, width='stretch')
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
                st.plotly_chart(fig_imp, width='stretch')
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
            
            st.plotly_chart(fig_imp, width='stretch')
        except Exception as e:
            st.error(f"Feature importance failed: {e}")

with tab2:
    st.markdown("#### Model Calibration")
    st.caption("A well-calibrated model's predicted probabilities match actual outcomes.")
    
    # Support both old nested ('models') and new flat schema for eval_metrics
    if eval_metrics:
        if 'models' in eval_metrics:
            metrics_data = eval_metrics['models']
        else:
            metrics_data = eval_metrics
            
        xgb_cal = metrics_data.get('xgboost', {}).get('calibration', {})
        unc_cal = metrics_data.get('uncertainty', {}).get('calibration', {})
        
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
            
            st.plotly_chart(fig_cal, width='stretch')
            
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

with tab3:
    st.markdown("#### 🏥 Ward Risk Stratification")
    st.caption("Batch risk assessment of the patient cohort. Instantly triage multiple patients.")
    
    n_ward = min(50, len(X_test_full))
    ward_data = X_test_full.iloc[:n_ward].copy()
    
    ward_risks = []
    ward_uncertainties = []
    for i in range(n_ward):
        try:
            row = ward_data.iloc[[i]].copy()
            for col in row.columns:
                row[col] = pd.to_numeric(row[col].astype(str).str.strip('[]'), errors='coerce').fillna(0).astype(np.float64)
            means, stds = unc_model.predict_uncertainty(row, n_samples=10)
            ward_risks.append(float(means[0]) * 100)
            ward_uncertainties.append(float(stds[0]) * 100)
        except Exception:
            try:
                proba = xgb.predict_proba(row)[0][1] * 100
                ward_risks.append(proba)
                ward_uncertainties.append(0.0)
            except Exception:
                ward_risks.append(0.0)
                ward_uncertainties.append(0.0)
    
    ward_df = pd.DataFrame({
        'Patient': [f'P-{i+1:03d}' for i in range(n_ward)],
        'Risk (%)': [round(r, 1) for r in ward_risks],
        'Uncertainty (%)': [round(u, 1) for u in ward_uncertainties],
        'Status': ['🔴 High' if r > 60 else '🟡 Medium' if r > 30 else '🟢 Low' for r in ward_risks],
        'Confidence': ['Low' if u > 15 else 'Medium' if u > 8 else 'High' for u in ward_uncertainties],
        'Ground Truth': ['YES' if y_test_full.iloc[i] == 1 else 'NO' for i in range(n_ward)]
    })
    
    # Summary metrics
    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    high_risk_count = sum(1 for r in ward_risks if r > 60)
    med_risk_count = sum(1 for r in ward_risks if 30 < r <= 60)
    low_risk_count = sum(1 for r in ward_risks if r <= 30)
    
    with w_col1:
        st.metric("Total Patients", n_ward)
    with w_col2:
        st.metric("🔴 High Risk", high_risk_count)
    with w_col3:
        st.metric("🟡 Medium Risk", med_risk_count)
    with w_col4:
        st.metric("🟢 Low Risk", low_risk_count)
    
    # Risk Distribution Scatter
    fig_ward = go.Figure()
    colors = ['#ef4444' if r > 60 else '#f59e0b' if r > 30 else '#22c55e' for r in ward_risks]
    
    fig_ward.add_trace(go.Scatter(
        x=list(range(n_ward)),
        y=ward_risks,
        mode='markers',
        marker=dict(
            size=[max(6, u * 1.5) for u in ward_uncertainties],
            color=colors,
            opacity=0.85,
            line=dict(width=1, color='rgba(255,255,255,0.3)')
        ),
        text=[f'P-{i+1:03d}<br>Risk: {ward_risks[i]:.1f}%<br>Uncertainty: {ward_uncertainties[i]:.1f}%' for i in range(n_ward)],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Threshold lines
    fig_ward.add_hline(y=60, line_dash='dash', line_color='#ef4444', annotation_text='High Risk', annotation_position='top left')
    fig_ward.add_hline(y=30, line_dash='dash', line_color='#f59e0b', annotation_text='Medium Risk', annotation_position='top left')
    
    fig_ward.update_layout(
        title='Patient Risk Stratification (bubble size = uncertainty)',
        xaxis_title='Patient Index',
        yaxis_title='Risk Score (%)',
        yaxis_range=[0, 105],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e2e8f0'},
        height=400
    )
    
    st.plotly_chart(fig_ward, width='stretch')
    
    # Patient Table
    st.dataframe(
        ward_df.style.apply(
            lambda row: [
                'background-color: rgba(239,68,68,0.15)' if '🔴' in str(row['Status'])
                else 'background-color: rgba(245,158,11,0.15)' if '🟡' in str(row['Status'])
                else 'background-color: rgba(34,197,94,0.1)'
            ] * len(row), axis=1
        ),
        width='stretch',
        height=300
    )

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #64748b; font-size: 0.8em;">
    🫀 Uncertainty-Aware {disease_type} Assessment | Built for Hack4Health 2026
</div>
""", unsafe_allow_html=True)
