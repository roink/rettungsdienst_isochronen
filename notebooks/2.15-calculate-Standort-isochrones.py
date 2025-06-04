#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import h5py
import joblib
import xarray as xr
import os
os.nice(20)

# ----------------------------------------------------------------------------------------
# 1. Load model and data
# ----------------------------------------------------------------------------------------
print("[LOG] Loading trained pipeline model...")
pipeline_model = joblib.load("../data/interim/best_RF-Eintreffzeit.pkl")
pipeline_model.named_steps['rf'].n_jobs = 30

print("[LOG] Loading original training data...")
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
train_data = pd.read_parquet("../data/interim/selected-data.parquet")[feature_columns]
cat_cols = train_data.select_dtypes(include="category").columns

# remove unused categories in each
for col in cat_cols:
    train_data[col] = train_data[col].cat.remove_unused_categories()

# ----------------------------------------------------------------------------------------
# 2. Define FZ coordinates and load precomputed data
# ----------------------------------------------------------------------------------------
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

print("[LOG] Loading precomputed distance/duration arrays from HDF5...")
precomputed_data = {}
lookup_path = "../data/interim/distance_duration_lookup.h5"
with h5py.File(lookup_path, "r") as h5f:
    for fz_name in fz_coordinates:
        ds_name = f"distance_duration/{fz_name.replace(' ', '_')}"
        if ds_name in h5f:
            precomputed_data[fz_name] = h5f[ds_name][:]  # shape (ny, nx, 2)
        else:
            print(f"Warning: {ds_name} not found in HDF5")

# ----------------------------------------------------------------------------------------
# 3. Define the grid
# ----------------------------------------------------------------------------------------
print("[LOG] Defining coordinate grid...")
n_points = 1000
min_lon, max_lon = 7.37586644151297, 7.598894722085717
min_lat, max_lat = 51.264836728429444, 51.41859378414624

diff_lon = max_lon - min_lon
diff_lat = max_lat - min_lat

lon_space = np.linspace(min_lon - diff_lon, max_lon + diff_lon, n_points)
lat_space = np.linspace(min_lat - diff_lat, max_lat + diff_lat, n_points)

# Flattened grid for assignment (we will repeat per sample)
grid_lon_flat = np.repeat(lon_space, n_points)
grid_lat_flat = np.tile(lat_space, n_points)
n_total = n_points * n_points

for fz, arr in precomputed_data.items():
    print(fz)

# ----------------------------------------------------------------------------------------
# 4. Pre-sample static DataFrame once
# ----------------------------------------------------------------------------------------
print("[LOG] Sampling static attributes from train_data...")
n_samples = 100
bigN = n_total * n_samples

print("Draw a big sample of attributes except FZ/distance/duration")
sample_df_base = train_data.sample(bigN, replace=True).reset_index(drop=True)
print("Assign grid coordinates")
sample_df_base["EINSATZORT_lon"] = np.repeat(grid_lon_flat, n_samples)
sample_df_base["EINSATZORT_lat"] = np.repeat(grid_lat_flat, n_samples)

# ----------------------------------------------------------------------------------------
# 5. Generate maps per FZ
# ----------------------------------------------------------------------------------------
def run_for_fz(fz_name, distdur_arr):
    print(f"[LOG] Generating map for {fz_name}…")
    # copy the pre-sampled base frame
    sample_df = sample_df_base.copy()
    sample_df["Standort FZ"] = fz_name

    # overwrite only distance/duration
    dist_flat = distdur_arr[:, :, 0].flatten()
    dur_flat  = distdur_arr[:, :, 1].flatten()

    # Now tile each *cell’s* dist/dur n_samples times,
    # but keep the order so we can reshape back into (n_points,n_points,n_samples)
    # by first repeating each cell’s value n_samples times
    dist_repeated = np.repeat(dist_flat, n_samples)
    dur_repeated  = np.repeat(dur_flat,  n_samples)

    sample_df["distance"] = dist_repeated
    sample_df["duration"] = dur_repeated

    # predict
    preds = pipeline_model.predict(sample_df)
    # mask invalid
    invalid = np.isnan(dist_repeated) | np.isnan(dur_repeated)
    preds[invalid] = np.nan

    # now reshape into (n_points, n_points, n_samples)
    preds_3d = preds.reshape(n_points, n_points, n_samples)
    print("  preds_3d.shape:", preds_3d.shape)
    # -> e.g. (1000, 1000, 100)

    # compute mean/std over the *samples* axis=2
    mean2d = np.nanmean(preds_3d, axis=2)
    std2d  = np.nanstd( preds_3d, axis=2)

    # sanity check
    print("  mean2d.shape:", mean2d.shape)  # -> (n_points, n_points)

    # save:
    ds = xr.DataArray(
        mean2d,
        dims=("y", "x"),
        coords={"y": lat_space, "x": lon_space},
        name="mean_prediction"
    ).to_dataset()

    comp = dict(zlib=True, complevel=4)
    encoding = {"mean_prediction": comp}

    fname = f"predictions_{fz_name}.nc"
    ds.to_netcdf(fname, engine="netcdf4", encoding=encoding)
    print(f"[LOG] Saved mean map for {fz_name} to {fname}")

unique_fz = train_data["Standort FZ"].unique()

for fz in unique_fz:
    arr = precomputed_data.get(fz)
    if arr is None:
        print(f"Warning: no precomputed data for {fz}, skipping.")
    else:
        run_for_fz(fz, arr)

sample_df_base.shape

# In[ ]:

