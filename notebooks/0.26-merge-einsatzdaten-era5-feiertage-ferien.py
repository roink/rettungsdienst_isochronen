#!/usr/bin/env python
# coding: utf-8

"""
Script to merge the combined Einsatz/Weather dataset with the Feiertage‑Ferien data.
The script:
  - Loads the merged Einsatz/Weather dataset from "../data/interim/combined_einsatz_weather_data.parquet".
  - Loads the Feiertage‑Ferien dataset from "../data/interim/Feiertage-Ferien.parquet".
  - Creates a date column (without time) in the Einsatz/Weather data by flooring ALARMZEIT.
  - Merges the two datasets based on the date.
  - Saves the resulting dataset to "../data/interim/combined_einsatz_weather_dates.parquet".
"""

import os
import pandas as pd

# Define file paths
base_dir = "../data/interim"
einsatz_weather_path = os.path.join(base_dir, "combined_einsatz_weather_data.parquet")
feiertage_path = os.path.join(base_dir, "Feiertage-Ferien.parquet")
output_path = os.path.join(base_dir, "combined_einsatz_weather_dates.parquet")

# Load the merged Einsatz/Weather dataset
df_einsatz_weather = pd.read_parquet(einsatz_weather_path)

# Load the Feiertage-Ferien dataset and ensure its date column is a datetime type
df_dates = pd.read_parquet(feiertage_path)
df_dates["date"] = pd.to_datetime(df_dates["date"])

# Create a new "date" column in the Einsatz/Weather data by flooring the timestamp.
if "ALARMZEIT" in df_einsatz_weather.columns:
    df_einsatz_weather["date"] = pd.to_datetime(df_einsatz_weather["ALARMZEIT"]).dt.floor("D")
elif "valid_time" in df_einsatz_weather.columns:
    df_einsatz_weather["date"] = pd.to_datetime(df_einsatz_weather["valid_time"]).dt.floor("D")
else:
    raise ValueError("Neither 'ALARMZEIT' nor 'valid_time' column found in the Einsatz/Weather dataset.")

# Display some rows for plausibility check
print("=== Einsatz/Weather Data (date extraction) ===")
print(df_einsatz_weather[["ALARMZEIT", "date"]].head(20))
print("\n=== Feiertage-Ferien Data ===")
print(df_dates.head(20))

# Merge the two datasets based on the "date" column
merged_df = pd.merge(df_einsatz_weather, df_dates, on="date", how="left")

print("\n=== Merged Data Sample ===")
# Display a sample of key columns from the merged dataset
print(merged_df[["ALARMZEIT", "date", "Feiertag", "Ferien"]].head(20))

# Save the merged dataset to a new Parquet file
merged_df.to_parquet(output_path, index=False)
print(f"\nMerged dataset saved to {output_path}")

# In[ ]:

