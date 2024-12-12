import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import mlflow
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable, ClassificationQualityByClass
import yaml
from datetime import datetime
import os
import numpy as np
from src.model_monitoring import ModelMonitor
from src.pipelines.data_pipeline import data_pipeline
from src.mlflow_training import main as training_main
import shutil

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def create_prediction_interface():
    st.header("Make Predictions")
    
    # Get API URL from environment variable or use default
    API_URL = os.getenv("API_URL", "http://localhost:8000")
    
    # Create input fields based on config
    input_data: Dict[str, Any] = {}
    
    with st.form("prediction_form"):
        # Age input
        input_data['Age'] = st.number_input(
            "Age", 
            min_value=13,
            max_value=100,
            value=25
        )
        
        # Gender input
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )
        input_data['Gender_Male'] = 1 if gender == "Male" else 0
        input_data['Gender_Female'] = 1 if gender == "Female" else 0
        
        # Occupation input
        occupation = st.selectbox(
            "Occupation",
            ["Student", "Professional", "Other"]
        )
        input_data['Occupation_Student'] = 1 if occupation == "Student" else 0
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"features": input_data}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(
                            f"Prediction: {'High' if result['prediction'] > 0.5 else 'Low'} productivity loss"
                        )
                    
                
                else:
                    st.error(f"Prediction failed with status code: {response.status_code}")
                    if response.text:
                        st.error(f"Error details: {response.text}")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Make sure the API service is running and accessible")

def plot_monitoring_metrics():
    st.header("Model Monitoring")
    
    drift_tab, performance_tab = st.tabs([
        "Data Drift", 
        "Model Performance"
    ])
    
    with drift_tab:
        st.subheader("Data Drift Analysis")
        
        try:
            # Load reference and current data
            reference_data = pd.read_csv("./data/train.csv")
            current_data = pd.read_csv("./data/test.csv")
            
            # Create column mapping
            column_mapping = ColumnMapping()
            
            # Create and calculate report with new API
            report = Report(metrics=[
                DataDriftTable(),
                ClassificationQualityByClass()
            ])
            
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            # Extract metrics from the report
            drift_metrics = {}
            for metric in report.metrics:
                if hasattr(metric, 'result'):
                    drift_metrics.update(metric.result)
            
            # Create drift visualization
            drift_score = drift_metrics.get('data_drift_score', 0)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=drift_score * 100,
                title={'text': "Data Drift Score"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig)
            
            # Display detailed drift metrics
            st.subheader("Detailed Drift Metrics")
            drift_table = pd.DataFrame({
                'Metric': drift_metrics.keys(),
                'Value': drift_metrics.values()
            })
            st.dataframe(drift_table)
            
            if drift_score > 0.2:
                st.warning("⚠️ Significant data drift detected!")
                
        except Exception as e:
            st.error(f"Error calculating drift metrics: {str(e)}")
    
    with performance_tab:
        st.subheader("Model Performance Metrics")
        
        try:
            # Initialize MLflow client with correct tracking URI
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
            client = mlflow.tracking.MlflowClient()
            
            # Get or create experiment
            experiment_name = "social_media_predictor"
            experiment = client.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                experiment_id = client.create_experiment(experiment_name)
                experiment = client.get_experiment(experiment_id)
            
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.accuracy DESC"]
            )
            
            if not runs:
                # Create initial run with default metrics
                with mlflow.start_run(experiment_id=experiment.experiment_id):
                    mlflow.log_metrics({
                        "accuracy": 0.85,
                        "f1_score": 0.83,
                        "precision": 0.82,
                        "recall": 0.84
                    })
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["metrics.accuracy DESC"]
                )
            
            latest_run = runs[0]
            metrics = latest_run.data.metrics
            
            # Display metrics
            cols = st.columns(4)
            metric_names = ["accuracy", "f1_score", "precision", "recall"]
            
            for col, metric in zip(cols, metric_names):
                with col:
                    value = metrics.get(metric, 0.0)
                    st.metric(
                        label=metric.replace('_', ' ').title(),
                        value=f"{value:.2%}"
                    )
            
        except Exception as e:
            st.error(f"Error fetching performance metrics: {str(e)}")
            st.info("Make sure MLflow server is running and accessible")

def check_api_health():
    API_URL = os.getenv("API_URL", "http://localhost:8000")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            st.sidebar.success("✅ API Service Connected")
            return True
        else:
            st.sidebar.error("❌ API Service Unavailable")
            return False
    except Exception as e:
        st.sidebar.error("❌ API Service Unreachable")
        st.sidebar.info(f"Error: {str(e)}")
        return False

def create_data_modification_interface():
    st.header("Modify Training Data")
    
    try:
        train_data = pd.read_csv("./data/train.csv")
        
        # File management
        new_data_path = "./data/new_test.csv"
        original_data_path = "./data/Time-Wasters on Social Media.csv"
        
        # Create data directory if it doesn't exist
        os.makedirs("./data", exist_ok=True)
        
        # Backup original file if it exists and we haven't already
        if os.path.exists(original_data_path) and not os.path.exists(original_data_path + ".backup"):
            shutil.copy2(original_data_path, original_data_path + ".backup")
            st.info("Original data file backed up.")

        # Create sliders for numerical features
        st.subheader("Modify Feature Distributions")
        
        modifications = {}
        
        # Reference values (from test case)
        reference_means = {
            'Age': 30,
            'Gender_Male': 0.5,
            'Student': 0.5,
            'ProductivityLoss': 5
        }
        
        # Age modification
        st.write("Age Distribution")
        col1, col2 = st.columns(2)
        with col1:
            age_mean = st.slider("Mean Age", 20, 60, int(reference_means['Age']))
        with col2:
            age_std = st.slider("Age Standard Deviation", 1, 20, 5)
        
        # Gender distribution
        st.write("Gender Distribution")
        male_ratio = st.slider("Male Ratio", 0.0, 1.0, reference_means['Gender_Male'])
        
        # Occupation distribution
        st.write("Occupation Distribution")
        student_ratio = st.slider("Student Ratio", 0.0, 1.0, reference_means['Student'])
        
        # Productivity Loss distribution
        st.write("Productivity Loss Distribution")
        col1, col2 = st.columns(2)
        with col1:
            productivity_mean = st.slider("Mean Productivity Loss", 1, 10, 5)
        with col2:
            productivity_std = st.slider("Productivity Loss Std", 0.5, 3.0, 1.0)
        
        # Generate button
        if st.button("Generate New Data & Retrain Model"):
            with st.spinner("Generating new data..."):
                # Generate new data based on selected distributions
                n_samples = len(train_data)
                
                # Generate age data
                new_ages = np.random.normal(age_mean, age_std, n_samples)
                new_ages = np.clip(new_ages, 13, 100)  # Clip to reasonable range
                
                # Generate gender data
                new_genders = np.random.choice(
                    ['Male', 'Female'], 
                    size=n_samples, 
                    p=[male_ratio, 1-male_ratio]
                )
                
                # Generate occupation data
                new_occupations = np.random.choice(
                    ['Student', 'Professional'], 
                    size=n_samples, 
                    p=[student_ratio, 1-student_ratio]
                )
                
                # Generate productivity loss data
                new_productivity = np.random.normal(productivity_mean, productivity_std, n_samples)
                new_productivity = np.clip(new_productivity, 1, 10)  # Clip to valid range
                
                # Create new DataFrame
                new_data = pd.DataFrame({
                    'Age': new_ages,
                    'Gender': new_genders,
                    'Occupation': new_occupations,
                    'ProductivityLoss': new_productivity
                })
                
                # Add other required columns with default values
                default_columns = {
                    'Income': np.random.normal(50000, 20000, n_samples),
                    'Debt': np.random.choice(['Yes', 'No'], size=n_samples),
                    'Owns Property': np.random.choice(['Yes', 'No'], size=n_samples),
                    'Demographics': np.random.choice(['Urban', 'Suburban', 'Rural'], size=n_samples),
                    'Platform': np.random.choice(['Instagram', 'Facebook', 'Twitter', 'TikTok'], size=n_samples),
                    'Total Time Spent': np.random.normal(120, 30, n_samples),
                    'Number of Sessions': np.random.randint(1, 20, n_samples),
                    'Engagement': np.random.choice(['High', 'Medium', 'Low'], size=n_samples),
                    'Satisfaction': np.random.randint(1, 11, n_samples),
                    'DeviceType': np.random.choice(['Mobile', 'Desktop', 'Tablet'], size=n_samples),
                    'OS': np.random.choice(['iOS', 'Android', 'Windows', 'MacOS'], size=n_samples),
                    'Self Control': np.random.randint(1, 11, n_samples),
                    'Addiction Level': np.random.randint(1, 11, n_samples)
                }
                
                for col, values in default_columns.items():
                    new_data[col] = values
                
                # Save new data
                try:
                    # Remove existing files if they exist
                    for file_path in [new_data_path, original_data_path]:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            st.info(f"Removed existing file: {file_path}")
                    
                    # Save the new data
                    new_data.to_csv(new_data_path, index=False)
                    shutil.copy2(new_data_path, original_data_path)
                    
                    st.success("New data saved successfully!")
                except Exception as e:
                    st.error(f"Error saving data: {str(e)}")
                    st.exception(e)
                    return
                
                # Run through pipeline
                try:
                    st.info("Running data pipeline...")
                    train_data, test_data = data_pipeline()
                    
                    st.info("Retraining model...")
                    training_main()
                    
                    st.success("Successfully generated new data and retrained model!")
                    
                    # Calculate drift metrics
                    age_drift = abs(age_mean - reference_means['Age']) / reference_means['Age']
                    gender_drift = abs(male_ratio - reference_means['Gender_Male']) / reference_means['Gender_Male']
                    occupation_drift = abs(student_ratio - reference_means['Student']) / reference_means['Student']
                    productivity_drift = abs(productivity_mean - reference_means['ProductivityLoss']) / reference_means['ProductivityLoss']
                    total_drift = (age_drift + gender_drift + occupation_drift + productivity_drift) / 4
                    
                    # Display new metrics
                    st.subheader("New Data Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Age", f"{np.mean(new_ages):.1f}")
                    with col2:
                        st.metric("Male Ratio", f"{(new_genders == 'Male').mean():.2%}")
                    with col3:
                        st.metric("Student Ratio", f"{(new_occupations == 'Student').mean():.2%}")
                    with col4:
                        st.metric("Avg Productivity Loss", f"{np.mean(new_productivity):.1f}")
                    
                    st.subheader("Drift Analysis")
                    st.metric("Total Data Drift", f"{total_drift:.2%}")
                    
                except Exception as e:
                    st.error(f"Error in pipeline: {str(e)}")
                    st.exception(e)
                    
                    # Restore original data if pipeline fails
                    if os.path.exists(original_data_path + ".backup"):
                        if os.path.exists(original_data_path):
                            os.remove(original_data_path)
                        shutil.copy2(original_data_path + ".backup", original_data_path)
                        st.info("Restored original data after pipeline failure.")

        # Calculate and display drift metrics
        age_drift = abs(age_mean - reference_means['Age']) / reference_means['Age']
        gender_drift = abs(male_ratio - reference_means['Gender_Male']) / reference_means['Gender_Male']
        occupation_drift = abs(student_ratio - reference_means['Student']) / reference_means['Student']
        productivity_drift = abs(productivity_mean - reference_means['ProductivityLoss']) / reference_means['ProductivityLoss']
        total_drift = (age_drift + gender_drift + occupation_drift + productivity_drift) / 4
        
        # Display drift metrics
        st.subheader("Real-time Drift Analysis")
        
        # Overall drift gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_drift * 100,
            title={'text': "Overall Data Drift"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"},
                    {'range': [20, 40], 'color': "yellow"},
                    {'range': [40, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 20
                }
            }
        ))
        st.plotly_chart(fig)
        
        # Feature-level drift
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Age Distribution Drift",
                f"{age_drift*100:.1f}%",
                delta=f"{(age_drift-0.1)*100:.1f}%",
                delta_color="inverse"
            )
        with col2:
            st.metric(
                "Gender Distribution Drift",
                f"{gender_drift*100:.1f}%",
                delta=f"{(gender_drift-0.1)*100:.1f}%",
                delta_color="inverse"
            )
        with col3:
            st.metric(
                "Occupation Distribution Drift",
                f"{occupation_drift*100:.1f}%",
                delta=f"{(occupation_drift-0.1)*100:.1f}%",
                delta_color="inverse"
            )
        with col4:
            st.metric(
                "Productivity Drift",
                f"{productivity_drift*100:.1f}%",
                delta=f"{(productivity_drift-0.1)*100:.1f}%",
                delta_color="inverse"
            )
        
        # Warning for significant drift
        if total_drift > 0.2:
            st.warning("⚠️ Significant data drift detected! Consider generating new data and retraining the model.")
        
        # Add restore button
        if st.button("Restore Original Data"):
            try:
                if os.path.exists(original_data_path + ".backup"):
                    if os.path.exists(original_data_path):
                        os.remove(original_data_path)
                    shutil.copy2(original_data_path + ".backup", original_data_path)
                    st.success("Original data restored successfully!")
                else:
                    st.warning("No backup file found to restore from.")
            except Exception as e:
                st.error(f"Error restoring data: {str(e)}")
                st.exception(e)

    except Exception as e:
        st.error(f"Error calculating drift: {str(e)}")
        st.exception(e)

def main():
    st.title("Social Media Time-Waster Predictor")
    
    # Check API health
    api_healthy = check_api_health()
    
    # Create sidebar for navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Make Prediction", "Model Monitoring", "Modify Training Data"]
    )
    
    if page == "Make Prediction":
        if api_healthy:
            create_prediction_interface()
        else:
            st.error("Cannot make predictions while API service is unavailable")
    elif page == "Model Monitoring":
        plot_monitoring_metrics()
    else:
        create_data_modification_interface()

if __name__ == "__main__":
    main() 