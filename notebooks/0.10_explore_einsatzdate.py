#!/usr/bin/env python
# coding: utf-8

import pandas as pd

# Load the dataset
file_path = "../data/raw/2024-10-01 einsatzdaten23_modified_export.CSV"
df = pd.read_csv(file_path, sep=';')

# Display the first few rows of the dataset to understand its structure
print("Dataset Head:")
print(df.head())

# Display column data types and non-null counts for initial inspection
print("\nData Info:")
print(df.info())

# Summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())

# Define columns that need to be parsed as dates and times
date_columns = [
    "MELDUNGSEINGANG", "ALARMZEIT", "Status 3", "Status 4",
    "Status 7", "Status 8", "Status 1", "Status 2"
]

# Define a safe converter for German decimal-comma format
def safe_float_conversion(value):
    if pd.isna(value) or value == '':
        return None  # or np.nan if preferred
    return float(value.replace(",", "."))

# Load data and apply decimal and date parsing
data = pd.read_csv(
    file_path,
    sep=';',  # Specify separator if needed
    parse_dates=date_columns,  # Automatically parse these columns as datetime
    converters={
        "EINSATZORT_X": safe_float_conversion,
        "EINSATZORT_Y": safe_float_conversion,
        "ZIELORT_X": safe_float_conversion,
        "ZIELORT_Y": safe_float_conversion
    }
)

print("Dataset Head:")
print(data.head())

data.describe()

data.info()

# Display rows where 'EINSATZNUMMER' is missing (NaN or empty)
invalid_einsatznummer = data[data['EINSATZNUMMER'].isna()]

# Show the results
print("Rows with missing 'EINSATZNUMMER':")
invalid_einsatznummer.head(100)

# Check if 'EINSATZSTICHWORT' is categorical and count unique values
unique_values = data['EINSATZSTICHWORT_1'].nunique()
value_counts = data['EINSATZSTICHWORT_1'].value_counts()

print(f"'EINSATZSTICHWORT_1' has {unique_values} unique values.")
print("\nValue Counts:")
print(value_counts)

# Check if 'EINSATZSTICHWORT' is categorical and count unique values
unique_values = data['EINSATZSTICHWORT'].nunique()
value_counts = data['EINSATZSTICHWORT'].value_counts()

print(f"'EINSATZSTICHWORT' has {unique_values} unique values.")
print("\nValue Counts:")
print(value_counts)

# Count the number of rows where the two columns are the same and where they differ
same_count = (data['EINSATZSTICHWORT'] == data['EINSATZSTICHWORT_1']).sum()
different_count = len(data) - same_count

print(f"Number of rows where 'EINSATZSTICHWORT' and 'EINSATZSTICHWORT_1' are the same: {same_count}")
print(f"Number of rows where they differ: {different_count}")

# Display examples where they are different
different_examples = data[data['EINSATZSTICHWORT'] != data['EINSATZSTICHWORT_1']].head(n=100)

print("\nExamples where 'EINSATZSTICHWORT' and 'EINSATZSTICHWORT_1' differ:")
print(different_examples[['EINSATZSTICHWORT', 'EINSATZSTICHWORT_1']])

not_identical = data[data['EINSATZSTICHWORT'] != data['EINSATZSTICHWORT_1']]
# Check if 'EINSATZSTICHWORT' is categorical and count unique values
unique_values = not_identical['EINSATZSTICHWORT'].nunique()
value_counts = not_identical['EINSATZSTICHWORT'].value_counts()

print(f"'EINSATZSTICHWORT' has {unique_values} unique values.")
print("\nValue Counts:")
print(value_counts)

# Convert columns to categorical data type
data['EINSATZSTICHWORT'] = data['EINSATZSTICHWORT'].astype('category')
data['EINSATZSTICHWORT_1'] = data['EINSATZSTICHWORT_1'].astype('category')

# Check if 'EINSATZSTICHWORT' is categorical and count unique values
unique_values = data['SONDERSIGNAL'].nunique()
value_counts = data['SONDERSIGNAL'].value_counts()

print(f"'SONDERSIGNAL' has {unique_values} unique values.")
print("\nValue Counts:")
print(value_counts)

# Filter rows where 'SONDERSIGNAL' is not "J" or "N"
invalid_sondersignal = data[~data['SONDERSIGNAL'].isin(['J', 'N'])]

# Display the rows with invalid 'SONDERSIGNAL' values
print("Rows where 'SONDERSIGNAL' is neither 'J' nor 'N':")
print(invalid_sondersignal)

data.dtypes

import numpy as np
# Replace any value in 'SONDERSIGNAL' that is not "J" or "N" with NaN
data['SONDERSIGNAL'] = data['SONDERSIGNAL'].apply(lambda x: x if x in ['J', 'N'] else np.nan)

# Convert 'SONDERSIGNAL' to boolean, with 'J' as True and 'N' as False
data['SONDERSIGNAL_BOOL'] = data['SONDERSIGNAL'].map({'J': True, 'N': False}).astype(pd.BooleanDtype())

data.dtypes

# Check if 'SONDERSIGNAL_BOOL' is categorical and count unique values
unique_values = data['SONDERSIGNAL_BOOL'].nunique()
value_counts = data['SONDERSIGNAL_BOOL'].value_counts()

print(f"'SONDERSIGNAL_BOOL' has {unique_values} unique values.")
print("\nValue Counts:")
print(value_counts)

data['SONDERSIGNAL_BOOL'].isnull().sum()

# Check if 'Standort FZ' is categorical and count unique values
unique_values = data['Standort FZ'].nunique()
value_counts = data['Standort FZ'].value_counts()

print(f"'Standort FZ' has {unique_values} unique values.")
print("\nValue Counts:")
print(value_counts)

data['Standort FZ'] = data['Standort FZ'].astype('category')

# Check if 'Einsatzmittel-Typ' is categorical and count unique values
unique_values = data['Einsatzmittel-Typ'].nunique()
value_counts = data['Einsatzmittel-Typ'].value_counts()

print(f"'Einsatzmittel-Typ' has {unique_values} unique values.")
print("\nValue Counts:")
print(value_counts)

data['Einsatzmittel-Typ'] = data['Einsatzmittel-Typ'].astype('category')

# Check if 'FUNKRUFNAME' is categorical and count unique values
unique_values = data['FUNKRUFNAME'].nunique()
value_counts = data['FUNKRUFNAME'].value_counts()

print(f"'FUNKRUFNAME' has {unique_values} unique values.")
print("\nValue Counts:")
print(value_counts)

data['FUNKRUFNAME'] = data['FUNKRUFNAME'].astype('category')

# Check if 'Bezeichnung Zielort' is categorical and count unique values
unique_values = data['Bezeichnung Zielort'].nunique()
value_counts = data['Bezeichnung Zielort'].value_counts()

print(f"'Bezeichnung Zielort' has {unique_values} unique values.")
print("\nValue Counts:")
print(value_counts)

data['Bezeichnung Zielort'] = data['Bezeichnung Zielort'].astype('category')

# Check if 'Standort FZ' is categorical and count unique values
unique_values = data['ZIELORT'].nunique()
value_counts = data['ZIELORT'].value_counts()

print(f"'ZIELORT' has {unique_values} unique values.")
print("\nValue Counts:")
print(value_counts)

data['ZIELORT'] = data['ZIELORT'].astype('category')

data.dtypes

data.describe()

from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)

# Function to parse and transform coordinates
def transform_coordinates(rechtswert, hochwert):

    # Transform the coordinates
    lon, lat = transformer.transform(rechtswert, hochwert)
    return pd.Series([lat, lon])

data[['EINSATZORT_lat', 'EINSATZORT_lon']] = data.apply(lambda row: transform_coordinates(row['EINSATZORT_X'], row['EINSATZORT_Y']), axis=1)

data[['EINSATZORT_lat', 'EINSATZORT_lon']].describe()

data[['ZIELORT_lat', 'ZIELORT_lon']] = data.apply(lambda row: transform_coordinates(row['ZIELORT_X'], row['ZIELORT_Y']), axis=1)

data[['EINSATZORT_lat', 'EINSATZORT_lon']].describe()

# In[ ]:

