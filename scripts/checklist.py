import subprocess
import sys
import os

def run_step(name, cmd):
    print(f"\n🔹 Checking {name}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {name} Passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} Failed")
        print(e.stderr)
        return False

def main():
    print("🛡️ Project Health Checklist")
    print("==========================")
    
    steps = [
        ("Unit Tests", "pytest"),
        ("Security (Safety)", "pip freeze | findstr mapie"),  # Check dependencies
        ("Validation Script", "python scripts/external_validation.py")
    ]
    
    # Set PYTHONPATH for scripts
    os.environ["PYTHONPATH"] = "."

    success = True
    for name, cmd in steps:
        if not run_step(name, cmd):
            success = False
            
    if success:
        print("\n🎉 All checks passed! Project is healthy.")
        sys.exit(0)
    else:
        print("\n⚠️ Some checks failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
