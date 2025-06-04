#!/usr/bin/env python
# coding: utf-8

import os
import sys
import h5py
import numpy as np
import pandas as pd
import openrouteservice
from tqdm.notebook import tqdm

# Adjust import path
sys.path.append(os.path.dirname(os.getcwd()))
import src.ors as ors

# Start local ORS Docker container 
ors.start()

# Initialize ORS Client on local Docker
client = openrouteservice.Client(base_url='http://localhost:8080/ors')

# Standorte and their coordinates
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

# Create the grid
min_lon, max_lon = 7.37586644151297, 7.598894722085717
min_lat, max_lat = 51.264836728429444, 51.41859378414624
n_points = 1000

diff_lon = max_lon - min_lon
diff_lat = max_lat - min_lat

lon_space = np.linspace(min_lon - diff_lon, max_lon + diff_lon, n_points)
lat_space = np.linspace(min_lat - diff_lat, max_lat + diff_lat, n_points)
lon_grid, lat_grid = np.meshgrid(lon_space, lat_space)

# Flatten the grid for easier chunking
flat_lons = lon_grid.flatten()
flat_lats = lat_grid.flatten()
total_grid_points = len(flat_lons)  # n_points * n_points

# Setup HDF5 output
output_path = "../data/interim/distance_duration_lookup.h5"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Helper function to generate chunks
def chunker(seq, size):
    """Yield successive chunks of size 'size' from list 'seq'."""
    for pos in range(0, len(seq), size):
        yield pos, seq[pos:pos+size]

#########################
# Matrix-based approach #
#########################

# We'll process each Standort. For each Standort, we do these steps:
# 1) Create an HDF5 dataset of shape (n_points, n_points, 2).
# 2) Break the flattened grid coords into manageable chunks.
# 3) For each chunk, call ORS matrix with 'sources=[0]' (the Standort) and
#    'destinations=[1..len(chunk)+1]' (the chunk).
# 4) Store results in the correct indices of the dataset.

chunk_size = 1 
with h5py.File(output_path, "w") as h5f:
    standorte_list = list(fz_coordinates.items())
    for fz_idx, (fz_name, (fz_lat, fz_lon)) in enumerate(tqdm(standorte_list, desc='Standorte')):
        dataset_name = f"distance_duration/{fz_name.replace(' ', '_')}"
        dset = h5f.create_dataset(dataset_name, (n_points, n_points, 2), dtype="float32")

        # Flatten grid arrays, then chunk them
        for start_idx, chunk_coords in chunker(list(zip(flat_lons, flat_lats)), chunk_size):
            locations = [[fz_lon, fz_lat]] + [list(xy) for xy in chunk_coords]
            destinations_idx = list(range(1, 1 + len(chunk_coords)))

            try:
                matrix_data = client.distance_matrix(
                    locations=locations,
                    profile="driving-car",
                    sources=[0],
                    destinations=destinations_idx,
                    metrics=["distance", "duration"],
                    optimized=False
                )
                distances = matrix_data["distances"][0]
                durations = matrix_data["durations"][0]

                # Store results
                for i, dist_m in enumerate(distances):
                    global_index = start_idx + i
                    row = global_index // n_points
                    col = global_index % n_points

                    # Handle None values safely
                    if dist_m is None or durations[i] is None or dist_m < 0:
                        dset[row, col, 0] = np.nan
                        dset[row, col, 1] = np.nan
                    else:
                        dset[row, col, 0] = dist_m / 1000.0   # Convert meters to km
                        dset[row, col, 1] = durations[i] / 60.0  # Convert seconds to minutes

            except openrouteservice.exceptions.ApiError as e:
                print(f"Matrix API error for {fz_name} at chunk {start_idx}: {e}")
                for i in range(len(chunk_coords)):
                    global_index = start_idx + i
                    row = global_index // n_points
                    col = global_index % n_points
                    dset[row, col, 0] = np.nan
                    dset[row, col, 1] = np.nan

print(f"Lookup table saved to: {output_path}")
