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

# Spalten aus dem DataFrame entfernen
data = data.drop(columns=columns_to_drop, errors="ignore")

print(data.info())

# Daten filtern: nur Zeilen mit Einsatzniveau >= 1 behalten
data = data[(data['SONDERSIGNAL'] == True) & 
            ((data['Einsatzniveau'].notna() & (data['Einsatzniveau'] >= 1)) | 
             (data['Einsatzniveau_1'].notna() & (data['Einsatzniveau_1'] >= 1)))]

# Spalte "Einsatzniveau" entfernen
data = data.drop(columns=['Einsatzniveau','Einsatzniveau_1','SONDERSIGNAL'])

print(data.info())

print("\nDataFrame info:")
print(data.info())
print("\nSummary statistics:")
print(data.describe(include="all"))

# Check for missing values
print("\nMissing values in data:")
print(data.isnull().sum())

data = data.dropna()

# Daten filtern: nur Zeilen mit Anfahrtszeit >= 0 behalten
data = data[data['Eintreffzeit'] >= 0]
data = data[data['Fahrzeit'] >= 0]

Q1 = data['Eintreffzeit'].quantile(0.25)
Q3 = data['Eintreffzeit'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[data['Eintreffzeit'] <= upper_bound]

print("\nDataFrame info:")
print(data.info())
print("\nSummary statistics:")
print(data.describe(include="all"))

# Check for missing values
print("\nMissing values in data:")
print(data.isnull().sum())

output_path = os.path.join("../data/interim", "selected-data.parquet")
data.to_parquet(output_path, index=False)

data["in_8min"] = data["Eintreffzeit"] <= 8
data["in_12min"] = data["Eintreffzeit"] <= 12

data["in_8min"].describe()

data["Eintreffzeit"].describe()

data["in_12min"].describe()

# In[ ]:

