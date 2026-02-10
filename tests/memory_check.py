import sys
import os
import joblib
import psutil
import gc
import time

def check_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Initial Memory: {check_memory():.2f} MB")

# Mock loading models to see footprint
print("Loading Models...")
try:
    load_start = check_memory()
    tabpfn = joblib.load('models/tabpfn_model.pkl')
    print(f"TabPFN Loaded. Memory: {check_memory():.2f} MB (+{check_memory() - load_start:.2f} MB)")
    
    xgb = joblib.load('models/xgb_model.pkl')
    print(f"XGBoost Loaded. Memory: {check_memory():.2f} MB")
    
    unc_model = joblib.load('models/unc_model.pkl')
    print(f"Uncertainty Loaded. Memory: {check_memory():.2f} MB")
    
except Exception as e:
    print(f"Failed to load models: {e}")

# Check for cleanup
print("Cleaning up...")
del tabpfn
del xgb
del unc_model
gc.collect()
print(f"After Cleanup: {check_memory():.2f} MB")
