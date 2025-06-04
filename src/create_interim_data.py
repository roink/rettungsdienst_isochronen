# create_interim_data.py

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from src.load import load_Hauskoordinaten

import pandas as pd
import xarray as xr
import re

import numpy as np
from pyproj import Transformer


# Define date columns once, since they're the same for both datasets
DATE_COLUMNS = [
    "MELDUNGSEINGANG", "ALARMZEIT", "Status 3", "Status 4",
    "Status 7", "Status 8", "Status 1", "Status 2"
]

def load_and_combine_Einsatzdaten(save_path="../data/interim"):
    """
    Load two datasets (2023 and 2018-22), combine them into a single DataFrame, and save the result.
    This is the only function intended to be called from outside this module.

    Parameters:
    save_path (str): Directory path to save the combined dataset.

    Returns:
    pd.DataFrame: Combined dataset.
    """

    dataset1 = load_Einsatzdaten_2023()
    dataset2 = load_Einsatzdaten_2018_22()

    combined_data = pd.concat([dataset1, dataset2], ignore_index=True)

    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, "combined_einsatz_data.parquet")
    combined_data.to_parquet(output_file)

    print(f"Combined dataset saved to: {output_file}")
    return combined_data

# -----------------------
# Private helper functions
# -----------------------

def _safe_float_conversion(value):
    """Convert German decimal-comma format to float, safely handling NaN or empty strings."""
    if pd.isna(value) or value == '':
        return None
    return float(value.replace(",", "."))

def _transform_coordinates(rechtswert, hochwert, transformer):
    """Transform coordinates from EPSG:32632 to EPSG:4326."""
    try:
        lon, lat = transformer.transform(rechtswert, hochwert)
        return pd.Series([lat, lon])
    except:
        return pd.Series([None, None])

def _load_dataset(file_path, 
                  date_columns, 
                  has_einsatzstichwort_1=False, 
                  has_bezeichnung_zielort=False, 
                  has_zielort=False):
    """
    Load a dataset from file_path and apply all common transformations.
    Flags control whether certain optional columns exist and should be processed.
    """

    transformer = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)

    data = pd.read_csv(
        file_path,
        sep=';',
        parse_dates=date_columns,
        dayfirst=True,
        converters={
            "EINSATZORT_X": _safe_float_conversion,
            "EINSATZORT_Y": _safe_float_conversion,
            "ZIELORT_X": _safe_float_conversion,
            "ZIELORT_Y": _safe_float_conversion
        }
    )

    # Convert datetime columns to minute precision
    for col in date_columns:
        if col in data.columns:
            data[col] = data[col].dt.floor('min')

    # Common type conversions
    if 'EINSATZSTICHWORT' in data.columns:
        data['EINSATZSTICHWORT'] = data['EINSATZSTICHWORT'].astype('category')

    if has_einsatzstichwort_1 and 'EINSATZSTICHWORT_1' in data.columns:
        data['EINSATZSTICHWORT_1'] = data['EINSATZSTICHWORT_1'].astype('category')

    if 'EINSATZNUMMER' in data.columns:
        data['EINSATZNUMMER'] = data['EINSATZNUMMER'].astype(pd.Int64Dtype())
    if 'HAUPTEINSATZNUMMER' in data.columns:
        data['HAUPTEINSATZNUMMER'] = data['HAUPTEINSATZNUMMER'].astype(pd.Int64Dtype())

    # Drop rows that are completely empty
    data = data.dropna(how='all')

    # Process SONDERSIGNAL
    if 'SONDERSIGNAL' in data.columns:
        data['SONDERSIGNAL'] = data['SONDERSIGNAL'].apply(lambda x: x if x in ['J', 'N'] else np.nan)
        data['SONDERSIGNAL'] = data['SONDERSIGNAL'].map({'J': True, 'N': False}).astype(pd.BooleanDtype())

    # Convert other columns to category if they exist
    for cat_col in ['Standort FZ', 'Einsatzmittel-Typ', 'FUNKRUFNAME']:
        if cat_col in data.columns:
            data[cat_col] = data[cat_col].astype('category')

    if has_bezeichnung_zielort and 'Bezeichnung Zielort' in data.columns:
        data['Bezeichnung Zielort'] = data['Bezeichnung Zielort'].astype('category')

    if has_zielort and 'ZIELORT' in data.columns:
        data['ZIELORT'] = data['ZIELORT'].astype('category')

    # Add transformed coordinates
    if 'EINSATZORT_X' in data.columns and 'EINSATZORT_Y' in data.columns:
        data[['EINSATZORT_lat', 'EINSATZORT_lon']] = data.apply(
            lambda row: _transform_coordinates(row['EINSATZORT_X'], row['EINSATZORT_Y'], transformer), axis=1
        )

    if 'ZIELORT_X' in data.columns and 'ZIELORT_Y' in data.columns:
        data[['ZIELORT_lat', 'ZIELORT_lon']] = data.apply(
            lambda row: _transform_coordinates(row['ZIELORT_X'], row['ZIELORT_Y'], transformer), axis=1
        )

    return data

def load_Einsatzdaten_2023():
    file_path = "../data/raw/2024-10-01 einsatzdaten23_modified_export.CSV"
    return _load_dataset(
        file_path=file_path,
        date_columns=DATE_COLUMNS,
        has_einsatzstichwort_1=True,
        has_bezeichnung_zielort=True,
        has_zielort=True
    )

def load_Einsatzdaten_2018_22():
    file_path = "../data/raw/2024-11-28 einsatzdaten18-22_modified_export.CSV"
    # This dataset does not have EINSATZSTICHWORT_1, Bezeichnung Zielort, ZIELORT
    return _load_dataset(
        file_path=file_path,
        date_columns=DATE_COLUMNS,
        has_einsatzstichwort_1=False,
        has_bezeichnung_zielort=False,
        has_zielort=False
    )


def extract_ERA5(raw_dir="../data/raw", interim_dir="../data/interim", force=False):
    """
    Extract ERA5 data from GRIB files in the raw directory and save as CSV files in the interim directory.

    Parameters:
    - raw_dir (str): Path to the directory containing the raw GRIB files.
    - interim_dir (str): Path to the directory where CSV files will be saved.
    - force (bool): If True, recreate the CSVs even if they already exist.
    """
    # Ensure the interim directory exists
    os.makedirs(interim_dir, exist_ok=True)

    # Define a regex pattern to validate file names
    file_pattern = re.compile(r"ERA5-\d{4}-\d{2}\.grib")

    # Iterate over all files in the raw directory
    for file_name in os.listdir(raw_dir):
        # Skip files that don't match the naming scheme
        if not file_pattern.match(file_name):
            continue

        # Extract year and month from the file name
        year, month = file_name.split("-")[1], file_name.split("-")[2].split(".")[0]

        # Define paths
        grib_path = os.path.join(raw_dir, file_name)
        csv_path = os.path.join(interim_dir, f"ERA5-{year}-{month}.csv")

        # Check if the CSV already exists
        if not force and os.path.exists(csv_path):
            print(f"File {csv_path} already exists. Skipping...")
            continue

        print(f"Processing {grib_path}...")

        # Open the GRIB file with xarray
        ds = xr.open_dataset(grib_path, engine="cfgrib", filter_by_keys={"edition": 1},indexpath='')

        # Extract time series for variables
        t2m_series = ds["t2m"].sel(latitude=51.36, longitude=7.46, method="nearest").values.flatten() - 273.15  # Convert to Celsius
        tp_series = ds["tp"].sel(latitude=51.36, longitude=7.46, method="nearest").values.flatten()
        sf_series = ds["sf"].sel(latitude=51.36, longitude=7.46, method="nearest").values.flatten()
        
        # Flatten valid times
        valid_times = ds["valid_time"].values.flatten()
        
        ds = xr.open_dataset(grib_path, engine="cfgrib", filter_by_keys={"edition": 2},indexpath='')
        snowc_series = ds["snowc"].sel(latitude=51.36, longitude=7.46, method="nearest").values.flatten()

        

        # Create a DataFrame
        df = pd.DataFrame({
            "valid_time": valid_times,
            "temperature_celsius": t2m_series,
            "total_precipitation": tp_series,
            "snow_cover": snowc_series,
            "snowfall": sf_series,
        })

        # Save the DataFrame to CSV
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")


def Hauskoordinaten_latlon():
    df = load_Hauskoordinaten()
    transformer = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)

    # Function to parse and transform coordinates
    def transform_coordinates(rechtswert_str, hochwert_str):
        # Replace comma with dot for decimal conversion
        rechtswert_str = str(rechtswert_str).replace(',', '.')
        hochwert_str = str(hochwert_str).replace(',', '.')

        # Extract zone number and easting value
        zone_number = int(rechtswert_str[:2])  # First two digits
        easting = float(rechtswert_str[2:])    # Remaining digits

        # Convert northing to float
        northing = float(hochwert_str)

        # Transform the coordinates
        lon, lat = transformer.transform(easting, northing)
        return pd.Series([lat, lon])

    df[['lat', 'lon']] = df.apply(lambda row: transform_coordinates(row['Rechtswert'], row['Hochwert']), axis=1)
    current_dir = os.path.dirname(os.getcwd())
    file_path = os.path.join(os.path.dirname(current_dir), "data", "interim", "Hauskoordinaten_latlon.csv")
    
    df.to_csv(file_path)
