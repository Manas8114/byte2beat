import os

files = [
    r"Data\Cardiac Failure\cardiac_failure_processed.csv",
    r"Data\Cardiac Failure\cardio_base.csv",
    r"Data\Heart Attack\heart_processed.csv",
    r"Data\ECG Timeseries\ecg_timeseries.csv"
]

for f in files:
    print(f"--- {f} ---")
    try:
        if os.path.exists(f):
            with open(f, 'r', encoding='utf-8') as file:
                for i in range(5):
                    print(file.readline().strip())
        else:
            print("File not found.")
    except Exception as e:
        print(f"Error reading {f}: {e}")
    print("\n")
