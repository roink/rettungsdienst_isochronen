#!/usr/bin/env python
# coding: utf-8

"""
Preprocess ERA5 GRIB files to create a clean, integrated weather dataset.
This script:
    - Loads each GRIB file from a raw directory.
    - Extracts temperature (t2m), total precipitation (tp), snowfall (sf), and snow cover (snowc) 
      for a target location.
    - Converts temperature from Kelvin to Celsius.
    - Combines the data from multiple GRIB files into a single DataFrame.
    - Sorts the data by date and time.
    - Converts cumulative precipitation and snowfall to hourly values.
    - Saves the integrated weather dataset.
"""

import os
import re
import xarray as xr
import pandas as pd
import numpy as np

# Set the target location (latitude, longitude)
TARGET_LAT = 51.36
TARGET_LON = 7.46

# Open the GRIB file with xarray
ds = xr.open_dataset("../data/raw/ERA5-2018-03.grib", engine="cfgrib", filter_by_keys={"edition": 1},indexpath='')

ds

# In[ ]:

def calculate_hourly_values(cumulative, time_array):
    """
    Given a cumulative series and its corresponding time array,
    calculate the hourly differences. If the timestamp indicates a reset 
    (here, we assume a reset when the hour equals 1), the cumulative value
    is taken as the hourly value.
    """
    times = pd.to_datetime(time_array)
    hourly = np.empty_like(cumulative, dtype=float)
    hourly[0] = np.nan  # No previous data to subtract

    for i in range(1, len(cumulative)):
        # Use the timestamp directly (DatetimeIndex supports direct indexing)
        if times[i].hour == 1:
            hourly[i] = cumulative[i]
        else:
            hourly[i] = cumulative[i] - cumulative[i - 1]
            
        # Ensure no negative values (handle precision issues)
        if hourly[i] < 0:
            hourly[i] = 0.0
    return hourly

def preprocess_weather_file(grib_path):
    """
    Process a single ERA5 GRIB file:
        - Open the GRIB file using cfgrib (using different editions).
        - Extract variables for the target location.
        - Create and return a DataFrame with raw cumulative data.
    
    Parameters:
        grib_path (str): Path to the GRIB file.
    
    Returns:
        pd.DataFrame: Processed weather data (without hourly differences).
    """
    # --- Open the GRIB file using edition 1 ---
    ds1 = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        filter_by_keys={"edition": 1},
        indexpath=''
    )
    
    # Select the nearest grid point and flatten the arrays.
    t2m = ds1["t2m"].sel(latitude=TARGET_LAT, longitude=TARGET_LON, method="nearest").values.flatten()
    tp = ds1["tp"].sel(latitude=TARGET_LAT, longitude=TARGET_LON, method="nearest").values.flatten()
    sf = ds1["sf"].sel(latitude=TARGET_LAT, longitude=TARGET_LON, method="nearest").values.flatten()
    d2m = ds1["d2m"].sel(latitude=TARGET_LAT, longitude=TARGET_LON, method="nearest").values.flatten()
    
    # Convert temperature from Kelvin to Celsius.
    t2m_celsius = t2m - 273.15
    d2m_celsius = d2m - 273.15
    
    # Get valid times (flattened)
    valid_times = ds1["valid_time"].values.flatten()
    
    # --- Open the GRIB file using edition 2 for snow cover ---
    ds2 = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        filter_by_keys={"edition": 2},
        indexpath=''
    )
    snowc = ds2["snowc"].sel(latitude=TARGET_LAT, longitude=TARGET_LON, method="nearest").values.flatten()
    
    # Create a DataFrame with the raw data
    df = pd.DataFrame({
        "valid_time": pd.to_datetime(valid_times),
        "temperature_celsius": t2m_celsius,
        "total_precipitation": tp,
        "snowfall": sf,
        "snow_cover": snowc,
        "dewpoint_temperature" : d2m_celsius
    })
    
    return df.dropna()

def preprocess_all_weather(raw_dir="../data/raw", 
                           output_file="../data/interim/ERA5_combined.csv", 
                           force=False):
    """
    Process all ERA5 GRIB files in the raw directory, combine them into one DataFrame,
    sort by valid time, compute hourly differences, and save the integrated weather dataset.
    
    Parameters:
        raw_dir (str): Directory where raw GRIB files are stored.
        output_file (str): Path to save the combined CSV.
        force (bool): If True, force reprocessing even if output_file exists.
    
    Returns:
        pd.DataFrame: The integrated weather dataset.
    """
    # If the combined file already exists and force is False, load and return it.
    if os.path.exists(output_file) and not force:
        print(f"Loading existing integrated dataset from {output_file}")
        return pd.read_csv(output_file, parse_dates=["valid_time"])
    
    # Define a regex pattern to select files like "ERA5-YYYY-MM.grib"
    file_pattern = re.compile(r"ERA5-\d{4}-\d{2}\.grib")
    processed_dfs = []
    
    for file_name in os.listdir(raw_dir):
        if file_pattern.match(file_name):
            grib_path = os.path.join(raw_dir, file_name)
            print(f"Processing {grib_path}...")
            df = preprocess_weather_file(grib_path)
            processed_dfs.append(df)
    
    if not processed_dfs:
        print("No valid GRIB files found in the raw directory.")
        return None
    
    # Combine all monthly DataFrames into one and sort by valid time.
    combined_df = pd.concat(processed_dfs, ignore_index=True)
    combined_df.sort_values("valid_time", inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    
    # Now compute hourly differences from the cumulative series for the entire dataset.
    combined_df["hourly_precipitation"] = calculate_hourly_values(
        combined_df["total_precipitation"].values,
        combined_df["valid_time"].values
    )
    combined_df["hourly_snowfall"] = calculate_hourly_values(
        combined_df["snowfall"].values,
        combined_df["valid_time"].values
    )
    
    # Save the integrated DataFrame as a CSV file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined weather dataset saved to {output_file}")
    
    return combined_df

weather_df = preprocess_all_weather(force=True)

df = preprocess_weather_file("../data/raw/ERA5-2018-02.grib")

df

# In[ ]:

