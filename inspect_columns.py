import joblib
import pandas as pd
import sys
import os

# Add src to path to avoid ModuleNotFoundError for Utils if needed (though pandas structures usually don't need it unless custom classes)
sys.path.append(os.path.abspath('src'))

try:
    X_test = joblib.load('models/X_test.pkl')
    print("Columns in X_test:")
    print(X_test.columns.tolist())
except Exception as e:
    print(f"Error: {e}")
