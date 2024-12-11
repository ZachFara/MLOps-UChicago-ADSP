import kagglehub
from pathlib import Path
import shutil
import os
import glob
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML

import requests
import pandas as pd

import subprocess

def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print(result.stdout)
    

def checkout_dataset_version(version: str) -> None:
    """
    Checkout a specific version of the dataset using git and DVC, affecting only DVC-tracked data.
    """
    assert version in ['v1', 'v2'], "Version must be 'v1' or 'v2'."

    version_to_commit = {
        'v1': 'b054eedec2d083210755dc3bb3eff6edafb4109d',
        'v2': 'fbecb17d0ebfaf8f588d2cbaf399e25c1d4dd5b8',
    }

    checkout_hash = version_to_commit[version]

    # Limit git checkout to DVC-related files only
    dvc_files = [file for file in os.listdir(".") if file.endswith(".dvc")] + [".dvc"]
    for file in dvc_files:
        run_command(["git", "checkout", checkout_hash, "--", file])

    # Checkout dataset version with DVC
    run_command(["dvc", "checkout", "--force"])
    
def load_data_version(version:str) -> pd.DataFrame:
    checkout_dataset_version(version)
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    return train, test


def format_and_invoke(df, scaler, endpoint="http://127.0.0.1:5007/invocations"):
    continuous_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[continuous_columns] = scaler.transform(df[continuous_columns])
    df = df.apply(pd.to_numeric, errors='coerce')
    payload = {
        "dataframe_split": {
            "columns": df.columns.tolist(),
            "data": df.values.tolist()
        }
    }
    response = requests.post(endpoint, json=payload)
    return response

import pickle
scaler = pickle.load(open('scaler.pkl', 'rb'))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def classification_metrics(y_true, y_pred):
    
    y_pred = (y_pred > .5).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    return accuracy, precision, recall, f1, roc_auc

def main(port=5007):
    
    train, test_v1 = load_data_version('v1')
    x_test = test_v1.drop(columns=['Productivity Lost'])
    y_test = test_v1['Productivity Lost']
    
    response = format_and_invoke(x_test, scaler, f"http://127.0.0.1:{port}/invocations")
    predictions_v1 = np.array(response.json()['predictions'])
    
    accuracy, precision, recall, f1, roc_auc = classification_metrics(y_test,predictions_v1)

    # Print the metrics
    print("Metrics for v1:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    train, test_v2 = load_data_version('v2')
    x_test = test_v2.drop(columns=['Productivity Lost'])
    y_test = test_v2['Productivity Lost']
    response = format_and_invoke(x_test, scaler, f"http://127.0.0.1:{port}/invocations")
    predictions_v2 = np.array(response.json()['predictions'])
    
    accuracy, precision, recall, f1, roc_auc = classification_metrics(y_test,predictions_v2)
    
    # Print the metrics
    print("Metrics for v2:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    current_data = test
    
    import os
    from evidently.report import Report
    from evidently.metrics import DataDriftTable, ClassificationQualityByClass
    from evidently import ColumnMapping
    from pathlib import Path

    cwd = Path(os.getcwd())

    column_mapping = ColumnMapping(
        target="Productivity Lost",  
        prediction="predictions"  
    )

    # These might have gotten set during previous iterations
    if 'predictions' in current_data.columns:
        current_data.drop('predictions', axis=1, inplace=True)
        
    if 'predictions' in reference_data.columns:
        reference_data.drop('predictions', axis=1, inplace=True)

    current_data["predictions"] = xgb_model.predict(current_data.drop('Productivity Lost', axis=1))
    reference_data['predictions'] = xgb_model.predict(reference_data.drop('Productivity Lost', axis=1))

    # Show the report in the Jupyter Notebook
    report = Report(metrics=[
        DataDriftTable(), 
        ClassificationQualityByClass()
    ])

    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    os.makedirs(cwd / 'reports', exist_ok=True)
    report.save_html(str(cwd / 'reports' / 'report_A.html'))
    report.show()
    
    
    
    
    
    

if __name__ == '__main__':
    main()