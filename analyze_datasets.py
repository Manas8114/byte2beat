import pandas as pd
import os

files = {
    "Cardiac Failure (Processed)": r"Data\Cardiac Failure\cardiac_failure_processed.csv",
    "Cardiac Failure (Base)": r"Data\Cardiac Failure\cardio_base.csv",
    "Heart Attack": r"Data\Heart Attack\heart_processed.csv",
    "ECG Timeseries": r"Data\ECG Timeseries\ecg_timeseries.csv"
}

for name, filepath in files.items():
    print("=" * 80)
    print(f"DATASET: {name}")
    print("=" * 80)
    
    if not os.path.exists(filepath):
        print("❌ File not found!\n")
        continue
    
    try:
        df = pd.read_csv(filepath)
        
        print(f"\n📊 SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
        
        print(f"\n📋 COLUMNS ({len(df.columns)}):")
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            null_count = df[col].isna().sum()
            print(f"  • {col:30s} | Type: {str(dtype):10s} | Non-null: {non_null:6d} | Null: {null_count}")
        
        print(f"\n📈 STATISTICS:")
        print(df.describe(include='all').to_string())
        
        print(f"\n🔍 SAMPLE DATA (first 3 rows):")
        print(df.head(3).to_string())
        
        print(f"\n✅ Missing values summary:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("  No missing values!")
        
        print("\n")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}\n")
