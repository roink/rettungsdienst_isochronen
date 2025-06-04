#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Load  combined weather dataset ---
df = pd.read_csv("../data/interim/ERA5_combined.csv")
df['valid_time'] = pd.to_datetime(df['valid_time'])
df = df.sort_values('valid_time')

# --- 2. Define a dictionary with variable labels and corresponding column names ---
variables = {
    "Temperatur [°C]": "temperature_celsius",
    "Gesamtniederschlag [m]": "total_precipitation",
    "Niederschlag [m/h]": "hourly_precipitation",
    "Gesamtschneefall [m]": "snowfall",
    "Schneefall [m/h]": "hourly_snowfall",
    "Schneebedeckung [%]": "snow_cover",
    "Taupunkt [°C]": "dewpoint_temperature"
}

# --- 3. Plot the complete time series for each variable ---

# Compute month boundaries (using the first day of each month)
month_starts = pd.to_datetime(
    df['valid_time'].dt.to_period('M').unique().astype(str)
)
output_dir = "../reports/figures/"

# Function to sanitize filenames
def sanitize_filename(filename):
    return filename.split(' ')[0]  # Take only the part before the first space

for title, col in variables.items():
    plt.figure(figsize=(6, 4))  # Create a new figure for each variable
    sanitized_title = sanitize_filename(title)  # Sanitize the title for filename
    plt.plot(df['valid_time'], df[col], marker='o', linestyle='', markersize=2, label=title)
    plt.title(sanitized_title)
    plt.xlabel("Zeit")
    plt.ylabel(title)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"0.22_{sanitized_title}.png"))  # Save plot
    plt.show()

# --- 4. Detailed view for a single day ---
single_day = df['valid_time'].dt.date.iloc[3000]
df_day = df[df['valid_time'].dt.date == single_day]

fig, axs = plt.subplots(3, 2, figsize=(18, 12))
axs = axs.flatten()

for ax, (title, col) in zip(axs, variables.items()):
    ax.plot(df_day['valid_time'], df_day[col], marker='o', linestyle='', markersize=5, label=title)
    ax.set_title(f"{title} on {single_day}")
    ax.set_xlabel("Time")
    ax.set_ylabel(title)
    ax.grid(True)
    ax.legend()
plt.tight_layout()
plt.show()

# --- 5. Inspect seams between monthly files ---
# For each month boundary, we plot a short window around the boundary.
unique_periods = df['valid_time'].dt.to_period('M').unique()

if len(unique_periods) > 1:
    for i in range(10):
        # Get the last day of the current month and the first day of the next month
        current_month = unique_periods[i].to_timestamp()
        next_month = unique_periods[i + 1].to_timestamp()
        
        # Define a window: two days before and after the boundary
        window_start = current_month + pd.offsets.MonthEnd(0) - pd.Timedelta(days=2)
        window_end = next_month + pd.Timedelta(days=2)
        
        df_window = df[(df['valid_time'] >= window_start) & (df['valid_time'] <= window_end)]
        
        fig, axs = plt.subplots(3, 2, figsize=(18, 12))
        axs = axs.flatten()
        
        for ax, (title, col) in zip(axs, variables.items()):
            ax.plot(df_window['valid_time'], df_window[col], marker='o', linestyle='-', markersize=3, label=title)
            ax.set_title(f"{title}\n{current_month.strftime('%Y-%m')} to {next_month.strftime('%Y-%m')}")
            ax.set_xlabel("Time")
            ax.set_ylabel(title)
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        plt.show()

# Define the window boundaries.
# We include all times on 2018-02-28 and 2018-03-01.
window_start = pd.Timestamp("2018-02-28")
# To include the entire day of March 1, we set the end time to the end of that day.
window_end = pd.Timestamp("2018-03-01 23:59:59")

# Extract the subset of the dataframe for the defined window.
df_window = df[(df['valid_time'] >= window_start) & (df['valid_time'] <= window_end)]

# Display the numbers in the subset
print("Weather data between 2018-02-28 and 2018-03-01 (inclusive):")
print(df_window.to_string(index=False))

output_dir = "../reports/figures/"

# Function to sanitize filenames
def sanitize_filename(filename):
    return filename.split(' ')[0]  # Take only the part before the first space

for title, col in variables.items():
    plt.figure(figsize=(9, 6))  # Create a new figure for each variable
    sanitized_title = sanitize_filename(title)  # Sanitize the title for filename
    plt.plot(df_window['valid_time'], df_window[col], marker='o', linestyle='-', markersize=2, label=title)
    plt.title(title)
    plt.xlabel("Zeit")
    plt.ylabel(title)
    plt.grid(True)
    #plt.savefig(os.path.join(output_dir, f"Tag-{sanitized_title}.png"))  # Save plot
    plt.show()

# In[ ]:

