#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from pyproj import Transformer
import folium
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from src.load import load_Hauskoordinaten

# Load the CSV file

df = load_Hauskoordinaten()
# Display the number of rows in the dataset
num_rows = df.shape[0]
print(f"Number of rows in the dataset: {num_rows}")

# Initialize the transformer for UTM Zone 32N to WGS84
transformer = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)

# Function to parse and transform coordinates
def transform_coordinates(rechtswert_str, hochwert_str):
    # Replace comma with dot for decimal conversion
    rechtswert_str = str(rechtswert_str).replace(',', '.')
    hochwert_str = str(hochwert_str).replace(',', '.')

    # Extract zone number and easting value
    zone_number = int(rechtswert_str[:2])  # First two digits
    easting = float(rechtswert_str[2:])    # Remaining digits

    # Convert northing to float
    northing = float(hochwert_str)

    # Transform the coordinates
    lon, lat = transformer.transform(easting, northing)
    return pd.Series([lat, lon])

df[['lat', 'lon']] = df.apply(lambda row: transform_coordinates(row['Rechtswert'], row['Hochwert']), axis=1)

# Display the first few rows of the updated DataFrame
print(df[['Rechtswert', 'Hochwert', 'lat', 'lon']].head())

df.to_csv("../data/interim/Hauskoordinaten_latlon.csv")

