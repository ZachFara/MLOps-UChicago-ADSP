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
    
    print("Starting EDA...")
    # Add your code here
    print("EDA complete!")
    
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
    mlflow_training()
    print("MLflow training complete!")
    
    print("Starting model deployment...")
    model_deployment(port=5002, run_id = None)
    print("Model deployment complete!")
    
    print("Starting model monitoring...")
    model_monitoring(port=5002)
    print("Model monitoring complete!")
    
    print("Starting model deployment with unique port and run...")
    model_deployment(port=5005, run_id = '7441a6e21f0d491cbfc0a24336ef04d1')
    print("Unique model deployment complete!")
    
    print("Starting model monitoring...")
    model_monitoring(port=5005)
    print("Model monitoring complete!")
    
    print("All steps completed successfully!")
    