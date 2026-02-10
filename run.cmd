@echo off
setlocal EnableDelayedExpansion

echo ========================================================
echo  DeepMind-Inspired Cardiac Risk Assessment - Launcher
echo  Project Excellence Upgrade (v2.0.0)
echo ========================================================

:: Check for key artifacts (now checks for eval_metrics to ensure new rigor)
if exist "models\eval_metrics.json" goto SKIP_TRAINING

:TRAIN_MODEL
echo [INFO] Evaluation metrics not found. 
echo [INFO] Running FULL experiment (Training + Calibration + Evaluation)...
echo.
python run_experiment.py

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Training failed. Please check the logs above.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Training complete. Models and metrics saved.
goto RUN_TESTS

:SKIP_TRAINING
echo [INFO] Models and Evaluation Metrics found. Skipping training.
echo        (Delete 'models' folder to force re-training)

:RUN_TESTS
echo.
echo ========================================================
echo  Running Quick Health Checks (Unit Tests)...
echo ========================================================
echo.

python -m pytest tests/ -v --tb=short

if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some tests failed. Dashboard might be unstable.
    timeout /t 5
) else (
    echo [SUCCESS] All systems go.
)

echo.
echo ========================================================
echo  Launching Clinician's Trust Cockpit (Streamlit)...
echo ========================================================
echo.

streamlit run dashboard.py
