#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import openrouteservice
import os

import sys

sys.path.append(os.path.dirname(os.getcwd()))

from src.load import load_Hauskoordinaten_latlon
import src.ors as ors

df = load_Hauskoordinaten_latlon()
ors.start()

# Initialize OpenRouteService client
client = openrouteservice.Client(base_url='http://localhost:8080/ors')

# Function to calculate distance and duration using OpenRouteService
def calculate_route(row, start_coords):
    end_coords = (row['lon'], row['lat'])
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

#  Feuer- und Rettungswache 1 (Mitte) 
start_coords = (7.462957640309114, 51.35746486686668)
# Apply the function to each row to calculate distance and duration
df[['distance_RW1', 'duration_RW1']] = df.apply(calculate_route, axis=1, start_coords=start_coords)

# Feuer- und Rettungswache 2 (Ost)
start_coords = (7.546339893752056, 51.3744605710539)
# Apply the function to each row to calculate distance and duration
df[['distance_RW2', 'duration_RW2']] = df.apply(calculate_route, axis=1, start_coords=start_coords)

# Rettungswache 3 (Vorhalle) 
start_coords = (7.4348827970092595, 51.38544480707952)
# Apply the function to each row to calculate distance and duration
df[['distance_RW3', 'duration_RW3']] = df.apply(calculate_route, axis=1, start_coords=start_coords)

# Rettungswache 4 (1) (HaTüWe) 
start_coords = (7.432389936920493, 51.35154303203356)
# Apply the function to each row to calculate distance and duration
df[['distance_RW4', 'duration_RW4']] = df.apply(calculate_route, axis=1, start_coords=start_coords)

# Rettungswache 5 (Süd) 
start_coords = (7.532746299850275, 51.3057632700336)
# Apply the function to each row to calculate distance and duration
df[['distance_RW5', 'duration_RW5']] = df.apply(calculate_route, axis=1, start_coords=start_coords)

df.to_csv("../data/interim/Routing_distanzen.csv")
