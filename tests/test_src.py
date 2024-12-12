import unittest
import pandas as pd
import numpy as np
import os
import mlflow
from pathlib import Path
import yaml
import json
import requests
from fastapi.testclient import TestClient
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable, ClassificationQualityByClass, ColumnDriftMetric, DatasetDriftMetric
import shutil

from src.api.main import app
from src.download_data import main as download_main
from src.preprocessing import main as preprocessing_main
from src.automl import main as automl_main
from src.mlflow_training import main as training_main
from src.model_deployment import main as deployment_main
from src.model_monitoring import ModelMonitor
from src.pipelines.data_pipeline import data_pipeline

class TestSocialMediaTimeWaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.client = TestClient(app)
        cls.data_dir = Path("./data")
        cls.data_dir.mkdir(exist_ok=True)
        
        # Create a small test dataset
        cls.test_data = pd.DataFrame({
            'Age': [25, 30, 35],
            'Gender': ['Male', 'Female', 'Male'],
            'Occupation': ['Student', 'Professional', 'Student'],
            'ProductivityLoss': [7, 3, 5]
        })
        cls.test_data.to_csv(cls.data_dir / "test_data.csv", index=False)
        
        # Clean up existing MLflow artifacts
        mlflow_dir = Path("mlruns/0/latest/artifacts/model")
        if mlflow_dir.exists():
            shutil.rmtree(mlflow_dir)
        mlflow_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Train and save a simple model for testing
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10)
        X = pd.DataFrame({
            'Age': [25, 30, 35],
            'Gender_Male': [1, 0, 1],
            'Gender_Female': [0, 1, 0],
            'Occupation_Student': [1, 0, 1]
        })
        y = pd.Series([1, 0, 1])
        model.fit(X, y)
        
        # Save model for API testing
        mlflow.sklearn.save_model(model, "mlruns/0/latest/artifacts/model")

    def test_1_download_data(self):
        """Test data download functionality"""
        try:
            download_main()
            self.assertTrue((self.data_dir / "Time-Wasters on Social Media.csv").exists())
        except Exception as e:
            self.fail(f"Data download failed with error: {str(e)}")

    def test_2_preprocessing(self):
        """Test data preprocessing"""
        try:
            preprocessing_main()
            self.assertTrue((self.data_dir / "train.csv").exists())
            self.assertTrue((self.data_dir / "test.csv").exists())
            
            # Verify preprocessing results
            train_df = pd.read_csv(self.data_dir / "train.csv")
            self.assertIn("Productivity Lost", train_df.columns)
            self.assertTrue(train_df["Productivity Lost"].isin([0, 1]).all())
        except Exception as e:
            self.fail(f"Preprocessing failed with error: {str(e)}")

    def test_3_data_pipeline(self):
        """Test the complete data pipeline"""
        try:
            # Check if input data exists
            input_file = self.data_dir / "Time-Wasters on Social Media.csv"
            if not input_file.exists():
                self.fail(f"Input data file not found at {input_file}")
            
            # Log input data state
            input_df = pd.read_csv(input_file)
            print(f"\nInput data shape: {input_df.shape}")
            print(f"Input data columns: {input_df.columns.tolist()}")
            
            # Run pipeline
            train_data, test_data = data_pipeline()
            
            # Detailed assertions with helpful messages
            self.assertIsInstance(train_data, pd.DataFrame, 
                                f"Expected train_data to be DataFrame, got {type(train_data)}")
            self.assertIsInstance(test_data, pd.DataFrame, 
                                f"Expected test_data to be DataFrame, got {type(test_data)}")
            
            print(f"\nTrain data shape: {train_data.shape}")
            print(f"Train data columns: {train_data.columns.tolist()}")
            print(f"Test data shape: {test_data.shape}")
            print(f"Test data columns: {test_data.columns.tolist()}")
            
            self.assertGreater(len(train_data), 0, 
                             f"Train data is empty: {train_data.shape}")
            self.assertGreater(len(test_data), 0, 
                             f"Test data is empty: {test_data.shape}")
            
            # Check for expected columns after preprocessing
            expected_columns = ['Age', 'Income', 'Debt', 'Gender_Female', 'Gender_Male', 
                              'Gender_Other']
            for col in expected_columns:
                self.assertIn(col, train_data.columns, 
                            f"Missing column {col} in train data")
                self.assertIn(col, test_data.columns, 
                            f"Missing column {col} in test data")
            
        except Exception as e:
            import traceback
            self.fail(f"Data pipeline failed with error: {str(e)}\n{traceback.format_exc()}")

    def test_4_model_training(self):
        """Test model training with MLflow"""
        try:
            training_main()
            # Verify MLflow experiment exists
            experiment = mlflow.get_experiment_by_name("Final Classification Experiment")
            self.assertIsNotNone(experiment)
        except Exception as e:
            self.fail(f"Model training failed with error: {str(e)}")

    def test_5_api_endpoints(self):
        """Test FastAPI endpoints"""
        try:
            # Test health check endpoint
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "healthy"})
            
            # Test prediction endpoint
            test_input = {
                "features": {
                    "Age": 25,
                    "Gender_Male": 1,
                    "Gender_Female": 0,
                    "Occupation_Student": 1
                }
            }
            response = self.client.post("/predict", json=test_input)
            self.assertEqual(response.status_code, 200, f"Prediction failed with response: {response.json()}")
            self.assertIn("prediction", response.json())
            self.assertIn("prediction_probability", response.json())
            
        except Exception as e:
            self.fail(f"API test failed with error: {str(e)}")

    def test_6_model_monitoring(self):
        """Test model monitoring functionality"""
        try:
            # Simple test to ensure Evidently is running
            simple_reference_data = pd.DataFrame({
                'feature': [1, 2, 3, 4, 5],
                'target': [0, 1, 0, 1, 0]
            })
            simple_current_data = pd.DataFrame({
                'feature': [2, 3, 4, 5, 6],
                'target': [1, 0, 1, 0, 1]
            })
            simple_report = Report(metrics=[DataDriftTable()])
            simple_report.run(reference_data=simple_reference_data, current_data=simple_current_data)
            simple_drift_metrics = simple_report.as_dict()
            print(f"Simple test drift metrics: {simple_drift_metrics}")

            # Ensure simple test produces metrics
            self.assertGreater(len(simple_drift_metrics), 0, "Evidently simple test failed to produce metrics")

            # Main test for model monitoring
            monitor = ModelMonitor(port=5000)
            
            # Create more realistic test data with proper columns and clear drift pattern
            np.random.seed(42)  # For reproducibility
            reference_data = pd.DataFrame({
                'Age': np.random.normal(30, 5, 1000),  # Increased sample size
                'Gender': np.random.choice(['Male', 'Female'], 1000, p=[0.5, 0.5]),
                'Occupation': np.random.choice(['Student', 'Professional'], 1000, p=[0.5, 0.5]),
                'ProductivityLoss': np.random.choice([0, 1], 1000, p=[0.6, 0.4])
            })
            
            # Create new data with significant drift
            new_data = pd.DataFrame({
                'Age': np.random.normal(40, 8, 1000),  # Changed mean and std
                'Gender': np.random.choice(['Male', 'Female'], 1000, p=[0.7, 0.3]),  # Changed proportions
                'Occupation': np.random.choice(['Student', 'Professional'], 1000, p=[0.3, 0.7]),
                'ProductivityLoss': np.random.choice([0, 1], 1000, p=[0.3, 0.7])
            })
            
            # Create column mapping for Evidently
            column_mapping = ColumnMapping(
                target="ProductivityLoss",
                numerical_features=['Age'],
                categorical_features=['Gender', 'Occupation'],
                prediction='prediction' if 'prediction' in new_data.columns else None
            )
            
            # Create report with specific metrics
            report = Report(metrics=[
                DataDriftTable(),
                ColumnDriftMetric(column_name="Age"),
                ColumnDriftMetric(column_name="Gender"),
                ColumnDriftMetric(column_name="Occupation"),
                DatasetDriftMetric()
            ])
            
            # Run report
            report.run(
                reference_data=reference_data,
                current_data=new_data,
                column_mapping=column_mapping
            )
            
            # Extract metrics differently
            report_dict = report.as_dict()
            print(f"\nFull report dictionary: {report_dict}")  # Debug print
            
            drift_metrics = {}
            if 'metrics' in report_dict:
                for metric_entry in report_dict['metrics']:
                    if 'result' in metric_entry:
                        print(f"\nProcessing metric: {metric_entry['metric']}")  # Debug print
                        drift_metrics.update(metric_entry['result'])
            
            # Specific assertions for drift detection
            self.assertIsInstance(drift_metrics, dict, "Drift metrics should be a dictionary")
            self.assertGreater(len(drift_metrics), 0, 
                              f"No drift metrics calculated. Available metrics: {drift_metrics}")
            
            # Test for specific metrics we expect
            expected_metrics = ['number_of_columns', 'number_of_drifted_columns', 'dataset_drift']
            for metric in expected_metrics:
                self.assertIn(metric, drift_metrics, f"Expected metric '{metric}' not found in results")
            
        except Exception as e:
            import traceback
            self.fail(f"Model monitoring failed with error: {str(e)}\n{traceback.format_exc()}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test artifacts"""
        # Clean up test data files
        test_files = [
            cls.data_dir / "test_data.csv",
        ]
        for file in test_files:
            if file.exists():
                file.unlink()

def main():
    unittest.main(verbosity=2)

if __name__ == "__main__":
    main() 