from src.download_data import main as download_data
from src.preprocessing import main as preprocessing
from src.automl import main as automl
from src.mlflow_training import main as mlflow_training
from src.model_deployment import main as model_deployment
from src.model_monitoring import main as model_monitoring

if __name__ == '__main__':
    print("Downloading data...")
    download_data()
    print("Data downloaded successfully!")
    
    print("Starting data preprocessing...")
    preprocessing()
    print("Data preprocessing complete!")
    
    print("Starting AutoML...")
    try:
        automl()
    except Exception as e:
        print("AutoML failed: It can be buggy sometimes")
    print("AutoML complete!")
    
    print("Starting MLflow training...")
    mlflow_training(num_iterations=2)
    print("MLflow training complete!")
    
    print("Starting model deployment...")
    model_deployment(port=5003, run_id = None)
    print("Model deployment complete!")
    
    import time
    time.sleep(5)
    
    print("Starting model monitoring...")
    model_monitoring(port=5003)
    print("Model monitoring complete!")
    
    print("Starting model deployment with unique port and run...")
    model_deployment(port=5005, run_id = '3c71046cec91494dbc4faf014bbfa2e4')
    print("Unique model deployment complete!")
    
    time.sleep(5)
    
    print("Starting model monitoring...")
    model_monitoring(port=5005)
    print("Model monitoring complete!")
    
    print("All steps completed successfully!")
    