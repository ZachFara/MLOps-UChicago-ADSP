import pandas as pd
import numpy as np
from typing import Dict, Any
import mlflow
from mlflow.tracking import MlflowClient
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable, ClassificationQualityByClass
from prometheus_client import start_http_server, Gauge, Counter
import yaml
import requests
import json
from datetime import datetime
import logging
from src.utils.logging_utils import setup_logging

# Initialize metrics
prediction_drift = Gauge('model_prediction_drift', 'Model prediction drift score')
data_drift = Gauge('model_data_drift', 'Data drift score')
model_accuracy = Gauge('model_accuracy', 'Model accuracy score')
prediction_latency = Gauge('model_prediction_latency', 'Model prediction latency')
prediction_errors = Counter('model_prediction_errors', 'Number of prediction errors')

logger = setup_logging(__name__)

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

class ModelMonitor:
    def __init__(self, port: int = 8000):
        self.config = load_config()
        self.reference_data = pd.read_csv("./data/train.csv")
        self.current_data = pd.DataFrame()
        self.predictions = []
        self.actuals = []
        self.port = port
        
    def start_monitoring_server(self, metrics_port: int = 8080):
        """Start Prometheus metrics server"""
        start_http_server(metrics_port)
        logger.info(f"Metrics server started on port {metrics_port}")
    
    def calculate_drift_metrics(self, new_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate drift metrics between reference and new data"""
        try:
            column_mapping = ColumnMapping()
            
            report = Report(metrics=[
                DataDriftTable(),
                ClassificationQualityByClass()
            ])
            
            report.run(
                reference_data=self.reference_data,
                current_data=new_data,
                column_mapping=column_mapping
            )
            
            # Extract metrics from the report
            drift_metrics = {}
            for metric in report.metrics:
                if hasattr(metric, 'result'):
                    drift_metrics.update(metric.result)
            
            # Update Prometheus metrics
            data_drift.set(drift_metrics.get('data_drift_score', 0))
            prediction_drift.set(drift_metrics.get('prediction_drift_score', 0))
            
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Error calculating drift metrics: {str(e)}")
            return {}
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate model performance metrics"""
        try:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            metrics = {
                'accuracy': accuracy_score(self.actuals, self.predictions),
                'f1': f1_score(self.actuals, self.predictions),
                'precision': precision_score(self.actuals, self.predictions),
                'recall': recall_score(self.actuals, self.predictions)
            }
            
            # Update Prometheus metrics
            model_accuracy.set(metrics['accuracy'])
            
            # Log to MLflow
            with mlflow.start_run(nested=True):
                for name, value in metrics.items():
                    mlflow.log_metric(f"monitoring_{name}", value)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def monitor_predictions(self, data: pd.DataFrame, prediction: Any, actual: Any = None):
        """Monitor individual predictions"""
        try:
            # Record prediction latency
            start_time = datetime.now()
            response = requests.post(
                f"http://localhost:{self.port}/predict",
                json={"features": data.to_dict(orient='records')[0]}
            )
            latency = (datetime.now() - start_time).total_seconds()
            prediction_latency.set(latency)
            
            if response.status_code != 200:
                prediction_errors.inc()
                logger.error(f"Prediction error: {response.text}")
                return
            
            # Update current data and predictions
            self.current_data = pd.concat([self.current_data, data])
            self.predictions.append(prediction)
            
            if actual is not None:
                self.actuals.append(actual)
            
            # Calculate drift if enough data is collected
            if len(self.current_data) >= self.config["monitoring"]["drift_detection"]["window_size"]:
                drift_metrics = self.calculate_drift_metrics(self.current_data)
                
                # Check if drift exceeds threshold
                if drift_metrics.get('data_drift_score', 0) > self.config["monitoring"]["alerts"]["drift_threshold"]:
                    logger.warning("Data drift detected! Consider retraining the model.")
                
                # Reset current data after drift calculation
                self.current_data = pd.DataFrame()
            
            # Calculate performance metrics if actuals are available
            if len(self.actuals) > 0:
                performance_metrics = self.calculate_performance_metrics()
                
                # Check if performance drops below threshold
                if performance_metrics.get('accuracy', 1) < self.config["monitoring"]["alerts"]["performance_threshold"]:
                    logger.warning("Model performance degradation detected! Consider retraining the model.")
            
        except Exception as e:
            logger.error(f"Error monitoring predictions: {str(e)}")
            prediction_errors.inc()

def main(port: int = 8000):
    """Main function to start model monitoring"""
    try:
        monitor = ModelMonitor(port=port)
        monitor.start_monitoring_server()
        logger.info("Model monitoring started successfully")
        
        # Keep the monitoring server running
        while True:
            pass
            
    except Exception as e:
        logger.error(f"Error in model monitoring: {str(e)}")
        raise

if __name__ == "__main__":
    main()