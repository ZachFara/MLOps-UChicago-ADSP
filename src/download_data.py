import kagglehub
import shutil
import os
import glob
import pandas as pd
import numpy as np

def main():
    os.makedirs('./data', exist_ok=True)
    path = kagglehub.dataset_download("muhammadroshaanriaz/time-wasters-on-social-media")

    # Print the download path and current working directory
    print("Downloaded path:", path)
    print("Current working directory:", os.getcwd())

    # Move all files found in the downloaded path to ./data
    for file in glob.glob(os.path.join(path, '**', '*.*'), recursive=True):
        if os.path.isfile(file):
            dest_path = os.path.join('./data', os.path.basename(file))
            # If file exists, remove it first
            if os.path.exists(dest_path):
                os.remove(dest_path)
            shutil.move(file, './data')

    # List the files in ./data to see what CSVs are actually there
    files_in_data = glob.glob('./data/*')
    print("Files in data directory:", files_in_data)

    # Try loading any CSV found
    csv_files = [f for f in files_in_data if f.lower().endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in ./data")

    df = pd.read_csv(csv_files[0])
    if 'UserID' in df.columns:
        df.drop('UserID', axis=1, inplace=True)

    df.head()
    
if __name__ == '__main__':
    main()