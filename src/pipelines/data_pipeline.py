from typing import Tuple, Dict, Any
import pandas as pd
import yaml
import os
from src.utils.logging_utils import setup_logging

logger = setup_logging(__name__)

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    try:
        with open("config/config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}", exc_info=True)
        raise

def download_data() -> pd.DataFrame:
    """Download data from source"""
    try:
        logger.info("Starting data download")
        from src.download_data import main as download_data_main
        download_data_main()
        
        data_path = "./data/Time-Wasters on Social Media.csv"
        if not os.path.exists(data_path):
            logger.error(f"Data file not found at {data_path}")
            raise FileNotFoundError(f"Data file not found at {data_path}")
            
        df = pd.read_csv(data_path)
        logger.info(f"Downloaded {len(df)} rows of data with columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}", exc_info=True)
        raise

def validate_data(df: pd.DataFrame) -> bool:
    """Validate data using basic pandas checks"""
    try:
        logger.info("Starting data validation")
        config = load_config()
        
        # Check minimum number of rows
        min_rows = config["data"]["validation"]["min_rows"]
        if len(df) < min_rows:
            logger.error(f"Data has fewer than {min_rows} rows: {len(df)}")
            return False
            
        # Define required columns based on the expected features
        required_columns = [
            'Age', 'Income', 'Debt', 'Owns Property', 'Total Time Spent',
            'Number of Sessions', 'Video ID', 'Video Length', 'Engagement',
            'Importance Score', 'Time Spent On Video', 'Number of Videos Watched',
            'Scroll Rate', 'Gender', 'Location', 'Profession', 'Demographics',
            'Platform', 'Video Category', 'Frequency', 'Watch Reason',
            'DeviceType', 'OS', 'Watch Time', 'CurrentActivity', 'ConnectionType'
        ]
        
        # Check required columns exist
        for column in required_columns:
            if column not in df.columns:
                logger.error(f"Required column missing: {column}")
                return False
                
        # Check for null values in required columns
        for column in required_columns:
            if df[column].isnull().any():
                logger.error(f"Null values found in column: {column}")
                return False
        
        logger.info("Data validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}", exc_info=True)
        raise

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess data"""
    try:
        logger.info("Starting data preprocessing")
        from src.preprocessing import main as preprocessing_main
        preprocessing_main()
        
        train_path = './data/train.csv'
        test_path = './data/test.csv'
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Preprocessed files not found")
        
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        
        logger.info(f"Preprocessed data - Train shape: {train.shape}, Test shape: {test.shape}")
        return train, test
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}", exc_info=True)
        raise

def data_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main data pipeline"""
    try:
        logger.info("Starting data pipeline")
        
        # Download data
        logger.info("Downloading data")
        raw_data = download_data()
        
        # Validate data
        logger.info("Validating data")
        is_valid = validate_data(raw_data)
        if not is_valid:
            raise ValueError("Data validation failed")
        
        # Preprocess data
        logger.info("Preprocessing data")
        train_data, test_data = preprocess_data(raw_data)
        
        logger.info("Pipeline completed successfully")
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise 