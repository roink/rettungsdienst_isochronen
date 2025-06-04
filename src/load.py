# load.py
from src.download import download_and_unzip
import pandas as pd
import os

current_dir = os.getcwd()
data_path = os.path.join(os.path.dirname(current_dir), "data")

def load_Hauskoordinaten():
    download_and_unzip('Hauskoordinaten')
    
    file_path = os.path.join(data_path, "raw", "Hauskoordinaten.csv")
    df = pd.read_csv(file_path,encoding='iso-8859-15', sep=';')
    columns_to_drop = ["Stand: 28.09.2024", "URL", "Unnamed: 17"]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    return df

def load_Hauskoordinaten_latlon():
    
    file_path = os.path.join(data_path, "interim", "Hauskoordinaten_latlon.csv")
    if not os.path.exists(file_path):
    
        import src.create_interim_data as create
        create.Hauskoordinaten_latlon()
    df = pd.read_csv(file_path)
    return df
    
def load_combined_einsatzdaten(force=False):
    """
    Load the combined Einsatzdaten dataset. If it does not exist, create it from the raw datasets.

    Returns:
        pd.DataFrame: The combined Einsatzdaten dataset.
    """
    combined_file_path = os.path.join(data_path, "interim", "combined_einsatz_data.parquet")
    
    # Check if the combined dataset exists
    if not os.path.exists(combined_file_path) or force:
        print("Combined dataset not found. Creating from raw datasets.")
        
        # Import previously defined function to combine datasets
        from src.create_interim_data import load_and_combine_Einsatzdaten
        load_and_combine_Einsatzdaten(os.path.join(data_path, "interim"))
        print("Combined dataset created.")
    else:
        print("Combined dataset already exists.")
    
    # Load and return the combined dataset
    print("Loading combined dataset...")
    df = pd.read_parquet(combined_file_path)
    print(f"Combined dataset loaded successfully. Shape: {df.shape}")
    return df
    
    
