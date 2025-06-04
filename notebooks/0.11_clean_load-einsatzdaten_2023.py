#!/usr/bin/env python
# coding: utf-8

import pandas as pd

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

file_path = "../data/raw/2024-10-01 einsatzdaten23_modified_export.CSV"

# Load data and apply decimal and date parsing
data = pd.read_csv(
    file_path,
    sep=';',  # Specify separator if needed
    parse_dates=date_columns,  # Automatically parse these columns as datetime
    dayfirst=True, 
    converters={
        "EINSATZORT_X": safe_float_conversion,
        "EINSATZORT_Y": safe_float_conversion,
        "ZIELORT_X": safe_float_conversion,
        "ZIELORT_Y": safe_float_conversion
    }
)

# Convert datetime columns to minute precision
for col in date_columns:
    data[col] = data[col].dt.floor('min')  # Floor to nearest minute

data['EINSATZSTICHWORT'] = data['EINSATZSTICHWORT'].astype('category')
data['EINSATZSTICHWORT_1'] = data['EINSATZSTICHWORT_1'].astype('category')
data['EINSATZNUMMER'] = data['EINSATZNUMMER'].astype(pd.Int64Dtype())
data['HAUPTEINSATZNUMMER'] = data['HAUPTEINSATZNUMMER'].astype(pd.Int64Dtype())

data = data.dropna(how='all')

import numpy as np
# Replace any value in 'SONDERSIGNAL' that is not "J" or "N" with NaN
data['SONDERSIGNAL'] = data['SONDERSIGNAL'].apply(lambda x: x if x in ['J', 'N'] else np.nan)

# Convert 'SONDERSIGNAL' to boolean, with 'J' as True and 'N' as False
data['SONDERSIGNAL'] = data['SONDERSIGNAL'].map({'J': True, 'N': False}).astype(pd.BooleanDtype())

data['Standort FZ'] = data['Standort FZ'].astype('category')
data['Einsatzmittel-Typ'] = data['Einsatzmittel-Typ'].astype('category')
data['FUNKRUFNAME'] = data['FUNKRUFNAME'].astype('category')
data['Bezeichnung Zielort'] = data['Bezeichnung Zielort'].astype('category')
data['ZIELORT'] = data['ZIELORT'].astype('category')

data.dtypes

from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)

# Function to parse and transform coordinates
def transform_coordinates(rechtswert, hochwert):

    # Transform the coordinates
    lon, lat = transformer.transform(rechtswert, hochwert)
    return pd.Series([lat, lon])

data[['EINSATZORT_lat', 'EINSATZORT_lon']] = data.apply(lambda row: transform_coordinates(row['EINSATZORT_X'], row['EINSATZORT_Y']), axis=1)
data[['ZIELORT_lat', 'ZIELORT_lon']] = data.apply(lambda row: transform_coordinates(row['ZIELORT_X'], row['ZIELORT_Y']), axis=1)

# In[ ]:

data.describe()

data.dtypes

data.info()

