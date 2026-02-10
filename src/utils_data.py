import pandas as pd
import numpy as np

def get_concept_map():
    """
    Returns a dictionary mapping 'Concept Name' to a list of feature names.
    Used for Concept-Bottleneck interpretability.
    """
    return {
        "Demographics": ["Age", "Sex_M", "Sex_F", "Height", "Weight"],
        "Vitals": ["RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Glucose"],
        "Lifestyle": ["Smoke", "Alcohol", "Active"],
        "Clinical": [
            "Oldpeak", 
            "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA", "ChestPainType_ASY", 
            "RestingECG_Normal", "RestingECG_ST", "RestingECG_LVH",
            "ExerciseAngina_Y", 
            "ST_Slope_Flat", "ST_Slope_Up", "ST_Slope_Down"
        ]
    }

def load_cardio_base(filepath):
    """
    Loads cardio_base.csv (semicolon delimited).
    Normalizes columns to match the 'Unified Schema'.
    """
    # Load with correct separator
    df = pd.read_csv(filepath, sep=';')
    
    # 1. Normalize Age (Days -> Years)
    df['Age'] = (df['age'] / 365.25).astype(int)
    
    # 2. Normalize Sex (1=Female, 2=Male in this dataset usually, need to verify or assume)
    # Mapping to Sex_M (0=Female, 1=Male) implies: if 2 is Male, then Sex_M = (val == 2)
    # Common Kaggle cardio dataset: 1: women, 2: men.
    df['Sex_M'] = (df['gender'] == 2).astype(int)
    
    # 3. Rename columns to Unified Schema
    # ap_hi -> RestingBP
    # cholesterol -> Cholesterol (keep ordinal 1,2,3 or normalize? keeping as is for tree models)
    # gluc -> Glucose (or match FastingBS?)
    df = df.rename(columns={
        'ap_hi': 'RestingBP',
        # 'cholesterol': 'Cholesterol', # Handle manually below to map Scale -> mg/dL
        'gluc': 'Glucose',

        'smoke': 'Smoke',
        'alco': 'Alcohol',
        'active': 'Active',
        'cardio': 'HeartDisease',
        'height': 'Height',
    })
    
    # 4. Map Cholesterol (Ordinal 1,2,3 -> Numeric mg/dL approx)
    # 1: Normal (<200), 2: Above Normal (200-239), 3: Well Above Normal (>240)
    # We map to representative means: 1->180, 2->225, 3->260
    # This aligns the semantic space with heart_processed.csv
    df['Cholesterol'] = df['cholesterol'].map({1: 180, 2: 225, 3: 260})
    
    # Drop raw columns
    df = df.drop(columns=['id', 'age', 'gender', 'ap_lo', 'cholesterol']) # ap_lo not in target schema for now, or could use
    
    return df

def load_heart_processed(filepath):
    """
    Loads heart_processed.csv and aligns to Unified Schema.
    """
    df = pd.read_csv(filepath)
    
    # Ensure booleans
    for col in df.columns:
        if df[col].dtype == 'bool' or df[col].astype(str).isin(['True', 'False']).any():
             df[col] = df[col].map({'True': 1, True: 1, 'False': 0, False: 0})
    
    return df

def load_and_preprocess_data(processed_path, base_path=None):
    """
    Loads datasets. If base_path is provided, it attempts to concatenate them
    into a 'Mega Dataset' for the Foundation Model approach.
    """
    # 1. Load Primary (High Fidelity)
    df_proc = load_heart_processed(processed_path)
    
    # 2. Load Secondary (High Volume), if requested
    if base_path:
        try:
            df_base = load_cardio_base(base_path)
            
            # Add missing columns to both (filling with NaN for imputation/TabPFN)
            # Identify union of columns
            all_cols = set(df_proc.columns).union(set(df_base.columns))
            
            # Align df_proc
            for c in all_cols:
                if c not in df_proc.columns:
                    df_proc[c] = np.nan
            
            # Align df_base
            for c in all_cols:
                if c not in df_base.columns:
                    df_base[c] = np.nan
            
            # Concatenate
            print(f"Merging datasets: {len(df_proc)} (Processed) + {len(df_base)} (Base)")
            df_final = pd.concat([df_proc, df_base], axis=0, ignore_index=True)
            
            # Fill NaNs? TabPFN can handle them, but XGBoost might need help or specific flag.
            # Improved imputation: Use median for numeric, mode for categorical.
            for col in df_final.columns:
                if df_final[col].dtype in ['float64', 'int64']:
                    df_final[col] = df_final[col].fillna(df_final[col].median())
                else:
                    df_final[col] = df_final[col].fillna(df_final[col].mode().iloc[0] if not df_final[col].mode().empty else 0) 
            
        except Exception as e:
            print(f"Warning: Could not merge cardio_base.csv: {e}")
            df_final = df_proc
    else:
        df_final = df_proc

    if 'HeartDisease' not in df_final.columns:
        raise ValueError("Target 'HeartDisease' missing.")
        
    y = df_final['HeartDisease']
    X = df_final.drop(columns=['HeartDisease'])
    
    return X, y, get_concept_map()
