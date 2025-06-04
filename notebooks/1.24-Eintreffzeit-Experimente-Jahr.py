#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import h5py
import joblib
import xarray as xr

from sklearn.neighbors import NearestNeighbors

# ----------------------------------------------------------------------------------------
# 1. -------------------- Load full training data and define features --------------------
# ----------------------------------------------------------------------------------------
print("[LOG] Loading original training data...")
full_data = pd.read_parquet("../data/interim/selected-data.parquet")
# Note: The training data must contain the 'Jahr' column.
# Define the feature columns used by the model.
feature_columns = [
    "Standort FZ",
           "EINSATZORT_lat",
           "EINSATZORT_lon",
           "Wochentag",
           "Feiertag",
           "Ferien",
           "Monat",
           "Uhrzeit",
           "distance",
           "duration",
           "temperature_celsius",
           "snow_cover",
           "dewpoint_temperature",
           "hourly_precipitation",
           "hourly_snowfall"
]

# ----------------------------------------------------------------------------------------
# 2. -------------------- Define FZ coordinates and load precomputed lookup data -----------
# ----------------------------------------------------------------------------------------
print("[LOG] Defining FZ coordinates...")
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

k = 10  # Number of neighbors

print("[LOG] Loading precomputed distance/duration arrays from HDF5...")
precomputed_data = {}
lookup_path = "../data/interim/distance_duration_lookup.h5"
with h5py.File(lookup_path, "r") as h5f:
    for fz_name in fz_coordinates.keys():
        dataset_name = f"distance_duration/{fz_name.replace(' ', '_')}"
        if dataset_name in h5f:
            precomputed_data[fz_name] = h5f[dataset_name][:]
        else:
            print(f"Warning: {dataset_name} not in HDF5, skipping...")

full_data.info()

# ----------------------------------------------------------------------------------------
# 3. -------------------- Define the grid and sampling parameters -------------------------
# ----------------------------------------------------------------------------------------
print("[LOG] Defining coordinate grid...")
n_points = 1000  # Must match the grid used when creating the HDF5 arrays
min_lon, max_lon = 7.37586644151297, 7.598894722085717
min_lat, max_lat = 51.264836728429444, 51.41859378414624

diff_lon = max_lon - min_lon
diff_lat = max_lat - min_lat

lon_space = np.linspace(min_lon - diff_lon, max_lon + diff_lon, n_points)
lat_space = np.linspace(min_lat - diff_lat, max_lat + diff_lat, n_points)
lon_grid, lat_grid = np.meshgrid(lon_space, lat_space)
grid_lat_flat = lat_grid.flatten()  # shape: (n_points^2,)
grid_lon_flat = lon_grid.flatten()  # shape: (n_points^2,)
n_total = len(grid_lat_flat)        # Total number of grid cells

n_samples = 100                   # Number of neighbor samples per grid cell
bigN = n_total * n_samples        # Total samples for the experiment

# ----------------------------------------------------------------------------------------
# 4. -------------------- Run base experiment for each year -------------------------------
# ----------------------------------------------------------------------------------------
years = sorted(full_data["Jahr"].unique())
print("[LOG] Running base experiment for each year:", years)

for year in years:
    print(f"\n[LOG] Processing year: {year}")
    
    # Subset training data for the current year and restrict to feature columns.
    data_year = full_data[full_data["Jahr"] == year].copy()
    data_year = data_year[feature_columns]
    print(f"[LOG] Data shape for year {year}: {data_year.shape}")
    
    # Load the year-specific tuned pipeline model.
    model_path = f"../data/interim/best_Eintreffzeit_{year}.pkl"
    print(f"[LOG] Loading model for year {year} from {model_path}...")
    pipeline_model = joblib.load(model_path)
  
    print("[LOG] Fit NearestNeighbors on the current year's training data (using lat/lon).")
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(data_year[['EINSATZORT_lat', 'EINSATZORT_lon']].values)
    print("[LOG] NearestNeighbors fitted on year's training data.")
    
    print("[LOG] Compute the nearest neighbors for every grid cell.")
    distances, indices = nn.kneighbors(np.column_stack([grid_lat_flat, grid_lon_flat]))
    
    print("[LOG] For each grid cell, randomly choose n_samples neighbor indices.")
    random_idx = np.random.randint(0, k, size=(n_total, n_samples))
    random_idx_flat = random_idx.ravel()
    row_idx = np.repeat(np.arange(n_total), n_samples)
    sampled_indices = indices[row_idx, random_idx_flat]
    
    print("[LOG] Retrieve the corresponding Standort FZ values from the year's training data.")
    fz_values = data_year["Standort FZ"].values
    fz_for_all_big = fz_values[sampled_indices]
    
    print("[LOG] Prepare arrays to hold the distance and duration values.")
    dist_array_big = np.empty(bigN, dtype=np.float32)
    dur_array_big  = np.empty(bigN, dtype=np.float32)
    
    print("[LOG] Determine grid indices for mapping into precomputed arrays.")
    i_array = np.arange(n_total) // n_points
    j_array = np.arange(n_total) % n_points
    i_big = np.repeat(i_array, n_samples)
    j_big = np.repeat(j_array, n_samples)
    
    print("[LOG] Fill the distance and duration arrays from the precomputed lookup.")
    unique_fz = np.unique(fz_for_all_big)
    for fz in unique_fz:
        if fz in precomputed_data:
            dist_dur_arr = precomputed_data[fz]
            mask = (fz_for_all_big == fz)
            idx_for_fz = np.where(mask)[0]
            dist_array_big[idx_for_fz] = dist_dur_arr[i_big[idx_for_fz], j_big[idx_for_fz], 0]
            dur_array_big[idx_for_fz]  = dist_dur_arr[i_big[idx_for_fz], j_big[idx_for_fz], 1]
        else:
            print(f"[WARNING] Precomputed data for {fz} not available; filling with NaN.")
            mask = (fz_for_all_big == fz)
            dist_array_big[mask] = np.nan
            dur_array_big[mask] = np.nan
            
    # ------------------------------------------------------------------------------------
    # Run the base experiment (no overrides)
    # ------------------------------------------------------------------------------------
    print(f"[LOG] Running base experiment for year {year}...")
    
    print("[LOG] Draw a large sample from the current year's training data.")
    sample_df = data_year.sample(bigN, replace=True).reset_index(drop=True)
    print("[LOG] Override columns with grid and precomputed values.")
    sample_df["Standort FZ"]    = fz_for_all_big
    sample_df["EINSATZORT_lat"] = np.repeat(grid_lat_flat, n_samples)
    sample_df["EINSATZORT_lon"] = np.repeat(grid_lon_flat, n_samples)
    sample_df["distance"]       = dist_array_big
    sample_df["duration"]       = dur_array_big
    
    print("[LOG] Use the year-specific model to predict.")
    preds_all = pipeline_model.predict(sample_df)
    
    print("[LOG] Mark grid samples with invalid distances/durations as NaN.")
    invalid_mask = (np.isnan(dist_array_big)) | (np.isnan(dur_array_big))
    preds_all[invalid_mask] = np.nan
    
    # Reshape predictions and compute the mean and standard deviation over the n_samples axis.
    preds_2d = preds_all.reshape(n_total, n_samples)
    preds_mean = np.nanmean(preds_2d, axis=1)
    preds_std  = np.nanstd(preds_2d, axis=1)
    
    # Reshape back to the grid shape.
    mean_2d = preds_mean.reshape(n_points, n_points)
    std_2d  = preds_std.reshape(n_points, n_points)
    
    # Create an xarray Dataset and save the output as NetCDF.
    ds = xr.Dataset(
        {
            "mean_prediction": (("y", "x"), mean_2d),
            "std_prediction": (("y", "x"), std_2d),
        },
        coords={
            "y": lat_space,
            "x": lon_space,
        },
    )
    
    comp = dict(zlib=True, complevel=4)
    encoding = {"mean_prediction": comp, "std_prediction": comp}
    outfile = f"predictions_base_{year}.nc"
    ds.to_netcdf(outfile, engine="netcdf4", encoding=encoding)
    print(f"[LOG] Saved NetCDF for year {year} to {outfile}")

print("[LOG] Base experiments complete for all years.")
