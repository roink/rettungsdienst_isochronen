#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.getcwd()))
import src.ors as ors

ors.start()

import openrouteservice
client = openrouteservice.Client(base_url='http://localhost:8080/ors')

import numpy as np
import pandas as pd
import h5py
import openrouteservice
import os

# Define the Start Standorte and their coordinates
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

# Define the grid of end locations
min_lon, max_lon = 7.37586644151297, 7.598894722085717
min_lat, max_lat = 51.264836728429444, 51.41859378414624
n_points = 1000  # Grid resolution

diff_lon = max_lon-min_lon
diff_lat = max_lat-min_lat

lon_space = np.linspace(min_lon-diff_lon, max_lon+diff_lon, n_points)
lat_space = np.linspace(min_lat-diff_lat, max_lat+diff_lat, n_points)
lon_grid, lat_grid = np.meshgrid(lon_space, lat_space)

from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import os
import h5py
import numpy as np
import openrouteservice

# Create HDF5 file to store results
output_path = "../data/interim/distance_duration_lookup.h5"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def compute_distance_duration(fz_name, fz_lat, fz_lon, i, j,pbar):
    """Compute distance and duration for a single grid point."""
    end_lat, end_lon = lat_grid[i, j], lon_grid[i, j]
    
    try:
        # Request route from OpenRouteService
        route = client.directions(
            coordinates=[(fz_lon, fz_lat), (end_lon, end_lat)],
            profile="driving-car",
            format="geojson"
        )
        
        # Extract distance (in km) and duration (in minutes)
        distance_km = route["features"][0]["properties"]["segments"][0]["distance"] / 1000
        duration_min = route["features"][0]["properties"]["segments"][0]["duration"] / 60
    
    except openrouteservice.exceptions.ApiError:
        distance_km, duration_min = np.nan, np.nan  # Mark errors as NaN
    pbar.update(1)
    
    return (i, j, distance_km, duration_min)

# Open HDF5 file for writing
with h5py.File(output_path, "w") as h5f, tqdm(total=len(fz_coordinates) * n_points**2, desc="Processing all locations") as pbar:
    
    for fz_name, (fz_lat, fz_lon) in fz_coordinates.items():
        dataset_name = f"distance_duration/{fz_name.replace(' ', '_')}"
        dset = h5f.create_dataset(dataset_name, (n_points, n_points, 2), dtype="float32")
        
        # Parallelize the computation
        results = Parallel(n_jobs=-1, backend="threading")(
            delayed(compute_distance_duration)(fz_name, fz_lat, fz_lon, i, j, pbar)
            for i in range(n_points) for j in range(n_points)
        )
        
        # Store results in HDF5 dataset
        for i, j, distance_km, duration_min in results:
            dset[i, j, 0] = distance_km
            dset[i, j, 1] = duration_min

print(f"Lookup table saved to: {output_path}")

import h5py

lookup_path = "../data/interim/distance_duration_lookup.h5"
fz_name = "Rettungswache Mitte"

# Open HDF5 file and retrieve dataset
with h5py.File(lookup_path, "r") as h5f:
    dataset_name = f"distance_duration/{fz_name.replace(' ', '_')}"
    distance_duration_data = h5f[dataset_name][:]  # Load the full array

print("Shape of stored distance/duration data:", distance_duration_data.shape)

import numpy as np
import h5py
import matplotlib.pyplot as plt

# Load the precomputed durations from the HDF5 file
lookup_path = "../data/interim/distance_duration_lookup.h5"

with h5py.File(lookup_path, "r") as h5f:
    for fz_name in fz_coordinates.keys():
        dataset_name = f"distance_duration/{fz_name.replace(' ', '_')}"
        
        if dataset_name not in h5f:
            print(f"Dataset {dataset_name} not found, skipping...")
            continue
        
        duration_data = h5f[dataset_name][..., 1]  # Extract only the duration (minutes)
        
        # Flatten data for scatter plot
        latitudes = lat_grid.flatten()
        longitudes = lon_grid.flatten()
        durations = duration_data.flatten()

        # Plot scatter
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(longitudes, latitudes, c=durations, cmap='viridis', marker='s', s=10, alpha=0.8)

        # Plot Standort location as red dot
        plt.scatter(*fz_coordinates[fz_name][::-1], color='red', marker='x', s=100, label="Standort")

        plt.colorbar(sc, label="Duration (min)")
        plt.title(f"Travel Duration from {fz_name}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.show()

# Load the precomputed durations from the HDF5 file
lookup_path = "../data/interim/distance_duration_lookup.h5"

with h5py.File(lookup_path, "r") as h5f:
    for fz_name in fz_coordinates.keys():
        dataset_name = f"distance_duration/{fz_name.replace(' ', '_')}"
        
        if dataset_name not in h5f:
            print(f"Dataset {dataset_name} not found, skipping...")
            continue
        
        duration_data = h5f[dataset_name][..., 0]  # Extract only the duration (minutes)
        
        # Flatten data for scatter plot
        latitudes = lat_grid.flatten()
        longitudes = lon_grid.flatten()
        durations = duration_data.flatten()

        # Plot scatter
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(longitudes, latitudes, c=durations, cmap='viridis', marker='s', s=10, alpha=0.8)

        # Plot Standort location as red dot
        plt.scatter(*fz_coordinates[fz_name][::-1], color='red', marker='x', s=100, label="Standort")

        plt.colorbar(sc, label="Distance (km)")
        plt.title(f"Travel Distance from {fz_name}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.show()

# In[ ]:

