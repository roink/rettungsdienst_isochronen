#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd

data = pd.read_parquet("../data/interim/combined_einsatz_weather_dates.parquet")
print("\n=== Merged Dataset ===")
print("Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nMerged DataFrame info:")
print(data.info())

# Extract the last character if it is preceded by a space and is a digit (0-3)
data['Einsatzniveau'] = data['EINSATZSTICHWORT'].str.extract(r' (\d)$')

# Convert the new column to numeric, so it becomes integers instead of strings
data['Einsatzniveau'] = data['Einsatzniveau'].astype('Int64')  # Supports NaN values

# Extract the last character if it is preceded by a space and is a digit (0-3)
data['Einsatzniveau_1'] = data['EINSATZSTICHWORT_1'].str.extract(r' (\d)$')

# Convert the new column to numeric, so it becomes integers instead of strings
data['Einsatzniveau_1'] = data['Einsatzniveau_1'].astype('Int64')  # Supports NaN values

data['Eintreffzeit'] = data['Status 4'] - data["MELDUNGSEINGANG"]
data['Eintreffzeit'] = pd.to_timedelta(data['Eintreffzeit']).dt.total_seconds() / 60

data['Fahrzeit'] = data['Status 4'] - data["ALARMZEIT"]
data['Fahrzeit'] = pd.to_timedelta(data['Fahrzeit']).dt.total_seconds() / 60

print("\nFirst 5 rows:")
print(data.head())

data['Wochentag'] = data['MELDUNGSEINGANG'].dt.strftime('%A').astype('category')
data['Monat'] = data['MELDUNGSEINGANG'].dt.strftime('%B').astype('category')

data['Uhrzeit'] = data['MELDUNGSEINGANG'].dt.hour + data['MELDUNGSEINGANG'].dt.minute / 60 + data['MELDUNGSEINGANG'].dt.second / 3600

print("\nMerged DataFrame info:")
print(data.info())

data['Standort FZ'] = data['Standort FZ'].astype('category')

data['Feiertag'] = data['Feiertag'].astype('boolean')
data['Ferien'] = data['Ferien'].astype('boolean')

print("\nMerged DataFrame info:")
print(data.info())

data["Standort FZ"] = data["Standort FZ"].astype(str).replace(
    {"Rettungswache Ost -NEF-": "Rettungswache Ost"}
).astype("category")

# Verify the changes
print(data["Standort FZ"].value_counts())

print("\nFirst 5 rows:")
print(data.head())

output_path = os.path.join("../data/interim", "all-data.parquet")
data.to_parquet(output_path, index=False)

