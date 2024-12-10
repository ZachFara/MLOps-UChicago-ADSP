from src.download_data import main as download_data
from src.preprocessing import main as preprocessing

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
    
    