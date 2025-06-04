#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd

data = pd.read_parquet("../data/interim/all-data.parquet")
print("\n=== Merged Dataset ===")
print("Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nMerged DataFrame info:")
print(data.info())

columns_to_drop = [
    "EINSATZNUMMER", "HAUPTEINSATZNUMMER", "MELDUNGSEINGANG", "ALARMZEIT","ALARMZEIT_floor",
    "Status 3", "Status 4", "Status 7", "Status 8", "Status 1", "Status 2",
    "EINSATZSTICHWORT_1", "EINSATZSTICHWORT", "EINSATZORT_X", "EINSATZORT_Y",
    "ZIELORT", "ZIELORT_X", "ZIELORT_Y", "Bezeichnung Zielort", "FUNKRUFNAME", "Einsatzmittel-Typ",
    "ZIELORT_lat", "ZIELORT_lon", "ZIELORT_ADRESSE", "FZ_lat", "FZ_lon",
    "ALARMZEIT_floor", "valid_time", "total_precipitation", "snowfall", "valid_time_floor", "date",
    "MELDUNGSEINGANG_floor"
]

print(data.info())

# Daten filtern: nur Zeilen mit Einsatzniveau >= 1 behalten
data = data[(data['SONDERSIGNAL'] == True) & 
            ((data['Einsatzniveau'].notna() & (data['Einsatzniveau'] >= 1)) | 
             (data['Einsatzniveau_1'].notna() & (data['Einsatzniveau_1'] >= 1)))]

print(data.info())

print("\nDataFrame info:")
print(data.info())
print("\nSummary statistics:")
print(data.describe(include="all"))

# Check for missing values
print("\nMissing values in data:")
print(data.isnull().sum())

# In[ ]:

# Daten filtern: nur Zeilen mit Anfahrtszeit >= 0 behalten
data = data[data['Eintreffzeit'] >= 0]
data = data[data['Fahrzeit'] >= 0]

print("\nDataFrame info:")
print(data.info())
print("\nSummary statistics:")
print(data.describe(include="all"))

# Check for missing values
print("\nMissing values in data:")
print(data.isnull().sum())

# Calculate Alarmverzögerung
data["Alarmverzögerung"] = data["Eintreffzeit"] - data["Fahrzeit"]

data = data.dropna(subset=["Alarmverzögerung"])

data.describe()

data

# Filter potential outliers
outliers = data[data["Alarmverzögerung"] > 60]
outliers

outliers.describe()

outliers.sort_values(by="Alarmverzögerung", ascending=False)

print(outliers.head())

outliers.to_csv("../data/interim/outliers.csv")

# Calculate additional percentiles
data["Alarmverzögerung"].quantile([0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995])

# In[ ]:

