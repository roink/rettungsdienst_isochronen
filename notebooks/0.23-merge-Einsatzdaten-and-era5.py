#!/usr/bin/env python
# coding: utf-8

"""
Script to load and explore the combined Einsatz/ORS dataset and weather data,
and then merge them based on timestamp.
"""

import os
import pandas as pd

# -------------------------------
# 1. Load and explore the Einsatz/ORS data
# -------------------------------
einsatz_path = os.path.join("../data/interim", "combined_einsatz_and_ORS_data.parquet")
einsatz_df = pd.read_parquet(einsatz_path)

print("=== Combined Einsatz/ORS Data ===")
print("Shape:", einsatz_df.shape)
print("\nFirst 5 rows:")
print(einsatz_df.head())
print("\nDataFrame info:")
print(einsatz_df.info())
print("\nSummary statistics:")
print(einsatz_df.describe(include="all"))

# Ensure that the event time column (MELDUNGSEINGANG) is in datetime format
einsatz_df["ALARMZEIT"] = pd.to_datetime(einsatz_df["ALARMZEIT"])

# Create a new column with the MELDUNGSEINGANG floored to the hour
einsatz_df["ALARMZEIT_floor"] = einsatz_df["ALARMZEIT"].dt.floor("h")

# Check for missing values
print("\nMissing values in Einsatz/ORS data:")
print(einsatz_df.isnull().sum())

# -------------------------------
# 2. Load and explore the weather data
# -------------------------------
weather_path = os.path.join("../data/interim", "ERA5_combined.csv")
weather_df = pd.read_csv(weather_path, parse_dates=["valid_time"])

print("\n=== Weather Data ===")
print("Shape:", weather_df.shape)
print("\nFirst 5 rows:")
print(weather_df.head())
print("\nDataFrame info:")
print(weather_df.info())
print("\nSummary statistics:")
print(weather_df.describe(include="all"))

# Although the weather data is typically hourly, we can floor its timestamps as well.
weather_df["valid_time_floor"] = weather_df["valid_time"].dt.floor("h")

# Check for missing values
print("\nMissing values in Weather data:")
print(weather_df.isnull().sum())

# -------------------------------
# 3. Merge the two datasets
# -------------------------------

# Sort both datasets by their time columns
einsatz_df.sort_values("ALARMZEIT", inplace=True)
weather_df.sort_values("valid_time", inplace=True)

# Use merge_asof to join the nearest weather record to each event.
merged_df = pd.merge(
    einsatz_df,
    weather_df,
    left_on="ALARMZEIT_floor",
    right_on="valid_time_floor",
    how="left",
    suffixes=("", "_weather")
)

print("\n=== Merged Dataset ===")
print("Shape:", merged_df.shape)
print("\nFirst 5 rows:")
print(merged_df.head())
print("\nMerged DataFrame info:")
print(merged_df.info())

print(merged_df[["ALARMZEIT", "ALARMZEIT_floor", "valid_time", "valid_time_floor"]].head(20))

# -------------------------------
# 4. Save the merged dataset
# -------------------------------
merged_output_path = os.path.join("../data/interim", "combined_einsatz_weather_data.parquet")
merged_df.to_parquet(merged_output_path, index=False)
print(f"\nMerged dataset saved to {merged_output_path}")

