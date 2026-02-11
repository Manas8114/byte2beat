"""
CLI entry point for UncertaintyML.

Commands:
    uncertaintyml train --data path/to/data.csv
    uncertaintyml predict --model models/ --input patient.json
    uncertaintyml serve --port 8000
"""

import argparse
import json
import sys
import os


def cmd_train(args):
    """Train models on a dataset."""
    from uncertaintyml.pipeline import UncertaintyPipeline, PipelineConfig
    from uncertaintyml.data import HeartDiseaseAdapter

    print("=" * 60)
    print("🫀 UncertaintyML — Training Pipeline")
    print("=" * 60)

    adapter = HeartDiseaseAdapter()
    X, y, concept_map = adapter.load(args.data, args.secondary)

    print(f"✅ Loaded {len(X)} samples, {X.shape[1]} features")

    models = args.models.split(",") if args.models else ["xgboost", "uncertainty"]
    config = PipelineConfig(
        models_to_train=models,
        uncertainty_epochs=args.epochs,
        output_dir=args.output,
    )

    pipe = UncertaintyPipeline(config)
    results = pipe.train(X, y, concept_map)
    pipe.save()

    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print(f"   Models saved to: {args.output}/")
    print("=" * 60)


def cmd_predict(args):
    """Predict risk for a patient from JSON input."""
    from uncertaintyml.pipeline import UncertaintyPipeline
    import pandas as pd

    pipe = UncertaintyPipeline.load(args.model_dir)

    if args.input:
        with open(args.input) as f:
            patient = json.load(f)
    else:
        print("Reading patient JSON from stdin...")
        patient = json.loads(sys.stdin.read())

    patient_df = pd.DataFrame([patient])
    result = pipe.predict(patient_df, model_name=args.model_name)

    print(json.dumps(result, indent=2))


def cmd_serve(args):
    """Start FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        print("❌ uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    os.environ["UNCERTAINTYML_MODELS_DIR"] = args.model_dir
    uvicorn.run("api.server:app", host=args.host, port=args.port, reload=args.reload)


def main():
    parser = argparse.ArgumentParser(
        prog="uncertaintyml",
        description="UncertaintyML — Uncertainty-Aware Medical Risk Assessment",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train
    p_train = subparsers.add_parser("train", help="Train models on a dataset")
    p_train.add_argument("--data", required=True, help="Path to primary CSV dataset")
    p_train.add_argument("--secondary", default=None, help="Path to secondary dataset for merging")
    p_train.add_argument("--models", default="xgboost,uncertainty", help="Comma-separated model names")
    p_train.add_argument("--epochs", type=int, default=100, help="Uncertainty model epochs")
    p_train.add_argument("--output", default="models", help="Output directory")

    # Predict
    p_pred = subparsers.add_parser("predict", help="Predict risk for a patient")
    p_pred.add_argument("--model-dir", default="models", help="Directory with saved models")
    p_pred.add_argument("--input", default=None, help="Path to patient JSON file (or stdin)")
    p_pred.add_argument("--model-name", default="uncertainty", help="Model to use for prediction")

    # Serve
    p_serve = subparsers.add_parser("serve", help="Start REST API server")
    p_serve.add_argument("--port", type=int, default=8000, help="Server port")
    p_serve.add_argument("--host", default="0.0.0.0", help="Server host")
    p_serve.add_argument("--model-dir", default="models", help="Directory with saved models")
    p_serve.add_argument("--reload", action="store_true", help="Enable hot reload")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
