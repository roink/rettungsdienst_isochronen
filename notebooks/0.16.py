#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from src.load import load_combined_einsatzdaten

data= load_combined_einsatzdaten()

data.info()

# Check if 'Standort FZ' is categorical and count unique values
unique_values = data['Standort FZ'].nunique()
value_counts = data['Standort FZ'].value_counts()

print(f"'Standort FZ' has {unique_values} unique values.")
print("\nValue Counts:")
print(value_counts)

# Mapping dictionary for 'Standort FZ' to coordinates
fz_coordinates = {
    "Rettungswache Mitte": (51.35738328512606, 7.463011574675159),
    "Rettungswache Ost": (51.37354069735159, 7.546357855336432),
    "Rettungswache Ost -NEF-": (51.37354069735159, 7.546357855336432),
    "Rettungswache Ev.Krhs. Haspe": (51.347649595944866, 7.410824948225933),
    "Rettungswache Vorhalle": (51.3857053741779, 7.4349754931565855),
    "Rettungswache Allgemeine Krankenhaus Hagen": (51.35392279941694, 7.466584396788664),
    "Rettungswache HaTüWe": (51.35146472534295, 7.432385861915431),
    "Rettungswache St. Johannes Hospital": (51.39586711277197, 7.473996499279848),
    "Rettungswache Dahl": (51.3055443277608, 7.53220077856884)
}

# Split the dictionary into two separate mappings for lat and lon
fz_lat_mapping = {key: coords[0] for key, coords in fz_coordinates.items()}
fz_lon_mapping = {key: coords[1] for key, coords in fz_coordinates.items()}

# Add the new columns using the mapping dictionaries
data['FZ_lat'] = data['Standort FZ'].map(fz_lat_mapping)
data['FZ_lon'] = data['Standort FZ'].map(fz_lon_mapping)

# Verify the new columns
print(data[['Standort FZ', 'FZ_lat', 'FZ_lon']].head())

data.head()

sondersignal_true_count = data[data['SONDERSIGNAL'] == True].shape[0]

sondersignal_true_non_null_fz_lon_count = data[(data['SONDERSIGNAL'] == True) & (data['FZ_lon'].notnull())].shape[0]

sondersignal_true_count

sondersignal_true_non_null_fz_lon_count

# Filter the data where SONDERSIGNAL is True and FZ_lon is null
filtered_data = data[(data['SONDERSIGNAL'] == True) & (data['FZ_lon'].isnull())]

# Check unique values and their counts for 'Standort FZ' in the filtered data
unique_values_filtered = filtered_data['Standort FZ'].nunique()
value_counts_filtered = filtered_data['Standort FZ'].value_counts()

unique_values_filtered, value_counts_filtered

# Calculate the number of unique combinations of the specified columns
unique_combinations_count = data[['FZ_lon', 'FZ_lat', 'EINSATZORT_lon', 'EINSATZORT_lat']].drop_duplicates().shape[0]

unique_combinations_count

import src.ors as ors

ors.start()

import openrouteservice
client = openrouteservice.Client(base_url='http://localhost:8080/ors')

# In[ ]:

import pandas as pd

# Extract unique combinations
unique_combinations = data[['FZ_lon', 'FZ_lat', 'EINSATZORT_lon', 'EINSATZORT_lat']].drop_duplicates()
unique_combinations_cleaned = unique_combinations.dropna(subset=['FZ_lon', 'FZ_lat', 'EINSATZORT_lon', 'EINSATZORT_lat'])

# Function to calculate distance and duration
def calculate_route(row):
    start_coords = (row['FZ_lon'], row['FZ_lat'])
    end_coords = (row['EINSATZORT_lon'], row['EINSATZORT_lat'])
    try:
        # Request directions
        route = client.directions(coordinates=[start_coords, end_coords], profile='driving-car', format='geojson')
        
        # Extract distance (in km) and duration (in minutes)
        distance_km = route['features'][0]['properties']['segments'][0]['distance'] / 1000  # Convert to km
        duration_min = route['features'][0]['properties']['segments'][0]['duration'] / 60  # Convert to minutes
        
        return pd.Series([distance_km, duration_min])
    except openrouteservice.exceptions.ApiError:
        # Handle any errors that occur during the API request
        return pd.Series([None, None])

# Apply the function to the cleaned unique combinations
unique_combinations_cleaned[['distance', 'duration']] = unique_combinations_cleaned.apply(calculate_route, axis=1)

# Merge the results back into the original dataset
data = data.merge(unique_combinations_cleaned, on=['FZ_lon', 'FZ_lat', 'EINSATZORT_lon', 'EINSATZORT_lat'], how='left')

os.makedirs("../data/interim", exist_ok=True)
output_file = os.path.join("../data/interim", "combined_einsatz_and_ORS_data.parquet")

data.to_parquet(output_file)

data.info()

# Check if 'Standort FZ' is categorical and count unique values
unique_values = data['EINSATZSTICHWORT'].nunique()
value_counts = data['EINSATZSTICHWORT'].value_counts()
pd.set_option('display.max_rows', None)
print(f"'EINSATZSTICHWORT' has {unique_values} unique values.")
print("\nValue Counts:")
print(value_counts)

pd.reset_option('display.max_rows')

# Extract the last character if it is preceded by a space and is a digit (0-3)
data['Einsatzniveau'] = data['EINSATZSTICHWORT'].str.extract(r' (\d)$')

# Convert the new column to numeric, so it becomes integers instead of strings
data['Einsatzniveau'] = data['Einsatzniveau'].astype('Int64')  # Supports NaN values

# Check the resulting DataFrame
print(data[['EINSATZSTICHWORT', 'Einsatzniveau']].head(100))  # Show only the first 100 rows

# Check if 'Standort FZ' is categorical and count unique values
unique_values = data['Einsatzniveau'].nunique()
value_counts = data['Einsatzniveau'].value_counts()
pd.set_option('display.max_rows', None)
print(f"'Einsatzniveau' has {unique_values} unique values.")
print("\nValue Counts:")
print(value_counts)

correlation_data = data.groupby('Einsatzniveau')['SONDERSIGNAL'].mean()
print(correlation_data)

# In[ ]:

