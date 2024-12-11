import kagglehub
import shutil
import os
import glob
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import load_model

def get_best_model(metric="test_f1", experiment_name="Final Classification Experiment"):
    """
    Retrieves the best model from the specified MLflow experiment based on the given metric.
    
    Args:
        metric (str): The metric to base the selection on (default is "test_f1").
        experiment_name (str): The name of the MLflow experiment.
    
    Returns:
        dict: Information about the best run and the loaded model.
    """
    client = MlflowClient()
    
    # Get the experiment ID by name
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    # Search runs within the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}' with metric '{metric}'.")
    
    best_run = runs[0]
    best_metric = best_run.data.metrics.get(metric, None)
    
    # Extract model URI and load the model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = load_model(model_uri)
    
    print(f"Best model found in experiment '{experiment_name}' with {metric}={best_metric}, run_id={best_run.info.run_id}")
    return {"model": model, "run_id": best_run.info.run_id, "metric": best_metric}

import subprocess

def serve_model(run_id, port):
    cmd = [
        "mlflow", "models", "serve",
        "-m", f"mlruns/150011855340888941/{run_id}/artifacts/model",
        "--port", str(port), "--env-manager=local"
    ]
    print("Running command:", " ".join(cmd))
    subprocess.Popen(cmd)


def main(port=5002, run_id = None):
    best_model_info = get_best_model(metric="f1_test", experiment_name="Final Classification Experiment")
    print(best_model_info)
    
    import mlflow.pyfunc
    if run_id is None:
        run_id = best_model_info['run_id']
    
    try:
        model = mlflow.pyfunc.load_model(f"mlruns/150011855340888941/{run_id}/artifacts/model")
        success = True
    except Exception as e:
        print(f"An error occurred: {e}")
        success = False
    
    serve_model(run_id, port)

    
    
if __name__ == '__main__':
    main()