#!/usr/bin/env python
# coding: utf-8

import xarray as xr

# Load the GRIB file with filter_by_keys
ds = xr.open_dataset(
    "../data/raw/ERA5-2023-01.grib",
    engine="cfgrib",
    filter_by_keys={"edition": 1},  # Specify edition: 1 for GRIB1 or 2 for GRIB2
    indexpath=''
)
print(ds)

import matplotlib.pyplot as plt

# Select the closest grid point
lat, lon = 51.36, 7.46
closest_point = ds.sel(latitude=lat, longitude=lon, method="nearest")

# Extract the time series for 't2m'
t2m_series = closest_point['t2m']  # Keep all the data points

# Convert temperature from Kelvin to Celsius
t2m_series_celsius = t2m_series - 273.15

# Flatten 'valid_time' and 't2m' values
valid_times = closest_point['valid_time'].values.flatten()
temperatures = t2m_series_celsius.values.flatten()

# Plot the continuous time series
plt.figure(figsize=(14, 8))
plt.plot(valid_times, temperatures, marker='o', linestyle='-', color='b')

# Format the plot
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Continuous Temperature Time Series")
plt.grid()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

import pandas as pd

# Flatten 'valid_time' and 't2m' values
valid_times = closest_point['valid_time'].values.flatten()
temperatures = t2m_series_celsius.values.flatten()

# Create a pandas DataFrame
df = pd.DataFrame({
    "valid_time": valid_times,
    "temperature_celsius": temperatures
})

# Display the DataFrame
print(df)

print(df.head(30))

# Extract the time series for 'sf' (snowfall)
sf_series = closest_point['sf']  # Keep all the data points

# Flatten 'valid_time' and 'sf' values
snowfall_values = sf_series.values.flatten()

# Add the snowfall data to the existing DataFrame
df["snowfall"] = snowfall_values

# Plot the snowfall time series
plt.figure(figsize=(14, 8))
plt.plot(valid_times, snowfall_values, marker='o', linestyle='-', color='b')

# Format the plot
plt.xlabel("Date")
plt.ylabel("Snowfall (meters)") 
plt.title("Continuous Snowfall Time Series")
plt.grid()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

# Display the updated DataFrame
print(df)

# Save to a CSV file (optional)
df.to_csv("../data/interim/temperature_snowfall_time_series.csv", index=False)

# Extract the time series for 'tp' (total precipitation)
tp_series = closest_point['tp']  # Keep all the data points

# Flatten 'valid_time' and 'tp' values
precipitation_values = tp_series.values.flatten()

# Add the precipitation data to the existing DataFrame
df["total_precipitation"] = precipitation_values

# Plot the total precipitation time series
plt.figure(figsize=(14, 8))
plt.plot(valid_times, precipitation_values, marker='o', linestyle='-', color='b')

# Format the plot
plt.xlabel("Date")
plt.ylabel("Total Precipitation (meters)")  # Units are typically meters; adjust if needed
plt.title("Continuous Total Precipitation Time Series")
plt.grid()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

# Display the updated DataFrame
print(df)

# Save to a CSV file (optional)
df.to_csv("../data/interim/temperature_snowfall_precipitation_time_series.csv", index=False)

print(df[23:80])

import numpy as np
def calculate_hourly_values(series, time_series):
    # Initialize an array to store hourly values
    hourly_values = np.zeros_like(series)

    # Iterate over the series and compute hourly values
    for i in range(1, len(series)):
        if time_series.iloc[i].hour == 1:  # Check for reset at 01:00:00
            hourly_values[i] = series[i]  # Reset detected, start fresh
        else:
            hourly_values[i] = series[i] - series[i - 1]

    # Ensure no negative values due to precision issues
    return hourly_values

# Apply to snowfall
df["hourly_snowfall"] = calculate_hourly_values(df["snowfall"], df["valid_time"])

# Apply to total precipitation
df["hourly_precipitation"] = calculate_hourly_values(df["total_precipitation"], df["valid_time"])

# Display the updated DataFrame
print(df[23:80])

import matplotlib.pyplot as plt

# Plot hourly snowfall
plt.figure(figsize=(14, 8))
plt.plot(df["valid_time"], df["hourly_snowfall"], marker='o', linestyle='-', color='blue', label="Hourly Snowfall")
plt.xlabel("Date")
plt.ylabel("Snowfall (meters of water equivalent)")
plt.title("Hourly Snowfall Time Series")
plt.grid()
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plot hourly precipitation
plt.figure(figsize=(14, 8))
plt.plot(df["valid_time"], df["hourly_precipitation"], marker='o', linestyle='-', color='green', label="Hourly Precipitation")
plt.xlabel("Date")
plt.ylabel("Precipitation (meters)")
plt.title("Hourly Precipitation Time Series")
plt.grid()
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Load the GRIB file with filter_by_keys
ds = xr.open_dataset(
    "../data/raw/ERA5-2023-01.grib",
    engine="cfgrib",
    filter_by_keys={"edition": 2},  # Specify edition: 1 for GRIB1 or 2 for GRIB2
    indexpath=''
)
print(ds)

# Extract snow cover data
snowc_series = ds["snowc"]

# Flatten the snow cover data and valid times
valid_times = ds["valid_time"].values.flatten()
snowc_values = snowc_series.sel(latitude=51.36, longitude=7.46, method="nearest").values.flatten()

# Add snow cover to the DataFrame
df["snow_cover"] = snowc_values

# Plot the snow cover time series
plt.figure(figsize=(14, 8))
plt.plot(valid_times, snowc_values, marker='o', linestyle='-', color='purple', label="Snow Cover")
plt.xlabel("Date")
plt.ylabel("Snow Cover (%)")  # Assuming snow cover is in percentage
plt.title("Snow Cover Time Series")
plt.grid()
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Display the updated DataFrame
print(df)

import os

import sys

sys.path.append(os.path.dirname(os.getcwd())) 

from src.create_interim_data import extract_ERA5

extract_ERA5(force=True)

# In[ ]:

