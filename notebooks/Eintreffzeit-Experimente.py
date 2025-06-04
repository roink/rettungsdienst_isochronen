# coding: utf-8

import numpy as np
import pandas as pd
import h5py
import joblib
import xarray as xr

from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

# ----------------------------------------------------------------------------------------
# 1. -------------------- Load model and data --------------------
# ----------------------------------------------------------------------------------------
print("[LOG] Loading trained pipeline model...")
# Load the entire pipeline (which includes the preprocessor and regressor)
pipeline_model = joblib.load("../data/interim/best_RF-Eintreffzeit.pkl")
pipeline_model.named_steps['rf'].n_jobs = 30

print("[LOG] Loading original training data...")
train_data = pd.read_parquet("../data/interim/selected-data.parquet")

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

train_data = train_data[feature_columns]
cat_cols = train_data.select_dtypes(include="category").columns
for col in cat_cols:
    train_data[col] = train_data[col].cat.remove_unused_categories()

# ----------------------------------------------------------------------------------------
# 2. -------------------- Define FZ coordinates and load precomputed data --------------------
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

k = 10
nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
print("[LOG] Fitting NearestNeighbors model on lat/lon of training data...")
nn.fit(train_data[['EINSATZORT_lat', 'EINSATZORT_lon']].values)

print("[LOG] Loading precomputed distance/duration arrays from HDF5...")
precomputed_data = {}
lookup_path = "../data/interim/distance_duration_lookup.h5"
with h5py.File(lookup_path, "r") as h5f:
    for fz_name in fz_coordinates.keys():
        dataset_name = f"distance_duration/{fz_name.replace(' ', '_')}"
        if dataset_name in h5f:
            arr = h5f[dataset_name][:]
            precomputed_data[fz_name] = arr
        else:
            print(f"Warning: {dataset_name} not in HDF5, skipping...")

# ----------------------------------------------------------------------------------------
# 3. -------------------- Define the grid --------------------
# ----------------------------------------------------------------------------------------
print("[LOG] Defining coordinate grid...")
n_points = 1000  # Must match how the HDF5 was created
min_lon, max_lon = 7.37586644151297, 7.598894722085717
min_lat, max_lat = 51.264836728429444, 51.41859378414624

diff_lon = max_lon - min_lon
diff_lat = max_lat - min_lat

lon_space = np.linspace(min_lon - diff_lon, max_lon + diff_lon, n_points)
lat_space = np.linspace(min_lat - diff_lat, max_lat + diff_lat, n_points)
lon_grid, lat_grid = np.meshgrid(lon_space, lat_space)

# Flatten for easier handling
grid_lat_flat = lat_grid.flatten()  # shape (n_points^2,)
grid_lon_flat = lon_grid.flatten()  # shape (n_points^2,)
n_total = len(grid_lat_flat)        # = n_points * n_points

# ----------------------------------------------------------------------------------------
# 4. -------------------- Precompute FZ, distance, duration for each grid cell ------------
# ----------------------------------------------------------------------------------------

print("[LOG] Precomputing FZ, distance, and duration arrays...")
n_samples = 100
coords = np.column_stack([grid_lat_flat, grid_lon_flat])
distances, indices = nn.kneighbors(coords)

random_idx = np.random.randint(0, k, size=(n_total, n_samples))
random_idx_flat = random_idx.ravel()

row_idx = np.repeat(np.arange(n_total), n_samples)
sampled_indices = indices[row_idx, random_idx_flat]

fz_values = train_data["Standort FZ"].values
fz_for_all_big = fz_values[sampled_indices]

bigN = n_total * n_samples
dist_array_big = np.empty(bigN, dtype=np.float32)
dur_array_big  = np.empty(bigN, dtype=np.float32)

i_array = np.arange(n_total) // n_points
j_array = np.arange(n_total) % n_points

i_big = np.repeat(i_array, n_samples)
j_big = np.repeat(j_array, n_samples)

for fz in np.unique(fz_for_all_big):
    dist_dur_arr = precomputed_data[fz]
    mask = (fz_for_all_big == fz)
    idx_for_fz = np.where(mask)[0]

    dist_array_big[idx_for_fz] = dist_dur_arr[i_big[idx_for_fz], j_big[idx_for_fz], 0]
    dur_array_big[idx_for_fz]  = dist_dur_arr[i_big[idx_for_fz], j_big[idx_for_fz], 1]

# ----------------------------------------------------------------------------------------
# Function to create the sample DataFrame, override columns, make predictions
# and return mean and std arrays (unshaped). We can then reshape to (n_points, n_points).
# ----------------------------------------------------------------------------------------
def run_experiment(overrides=None, outfile="experiment_output.nc"):
    """
    overrides: dict of {column_name: value_to_force}
       - special case for 'Wochentag' = 'weekday_random': random Monday..Friday
       - special case for 'Uhrzeit' that is a single float
    outfile: where to store the resulting netCDF

    Returns:
       preds_mean, preds_std  (both are shape (n_points * n_points,))
    """
    print(f"[LOG] Starting experiment for output file: {outfile}")
    if overrides:
        print(f"[LOG] Overrides in effect: {overrides}")
    else:
        print("[LOG] No overrides, base experiment.")

    # Step 5. ---------------- Build the big sample DataFrame ---------------
    print("[LOG] Sampling from train_data and assigning precomputed arrays...")
    sample_df = train_data.sample(bigN, replace=True).reset_index(drop=True)

    sample_df["Standort FZ"]    = fz_for_all_big
    sample_df["EINSATZORT_lat"] = np.repeat(grid_lat_flat, n_samples)
    sample_df["EINSATZORT_lon"] = np.repeat(grid_lon_flat, n_samples)
    sample_df["distance"]       = dist_array_big
    sample_df["duration"]       = dur_array_big

    # Apply overrides, if any
    if overrides is not None:
        print("[LOG] Applying overrides...")
        for col, val in overrides.items():
            if col == "Wochentag" and val == "weekday_random":
                print("[LOG] Randomly assigning Monday..Friday to Wochentag...")
                sample_df[col] = np.random.choice(
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    size=len(sample_df),
                )
            else:
                sample_df[col] = val
                
    print("[LOG] Making predictions...")
    preds_all = pipeline_model.predict(sample_df)

    print("[LOG] Marking invalid cells with NaN...")
    invalid_mask = (np.isnan(dist_array_big)) | (np.isnan(dur_array_big))
    preds_all[invalid_mask] = np.nan

    preds_2d = preds_all.reshape(n_total, n_samples)

    print("[LOG] Computing mean and std across samples...")
    preds_mean = np.nanmean(preds_2d, axis=1)
    preds_std  = np.nanstd(preds_2d, axis=1)

    mean_2d = preds_mean.reshape(n_points, n_points)
    std_2d  = preds_std.reshape(n_points, n_points)

    print("[LOG] Creating xarray Dataset and saving to NetCDF...")
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
    encoding = {
        "mean_prediction": comp,
        "std_prediction": comp
    }

    ds.to_netcdf(outfile, engine="netcdf4", encoding=encoding)
    print(f"[LOG] Saved NetCDF to {outfile}")
    return preds_mean, preds_std

# ----------------------------------------------------------------------------------------
# 5. -------------------- Run experiments ------------------------------------------------
# ----------------------------------------------------------------------------------------
print("[LOG] Running base experiment...")
run_experiment(overrides=None, outfile="predictions_base.nc")

if True:

    print("[LOG] Running experiment: Feiertag = True")
    run_experiment(overrides={"Feiertag": True}, outfile="predictions_feiertag_true.nc")

    print("[LOG] Running experiment: Feiertag = False")
    run_experiment(overrides={"Feiertag": False}, outfile="predictions_feiertag_false.nc")

    print("[LOG] Running experiment: Ferien = False")
    run_experiment(overrides={"Ferien": False}, outfile="predictions_ferien_false.nc")

    print("[LOG] Running experiment: Ferien = True")
    run_experiment(overrides={"Ferien": True}, outfile="predictions_ferien_true.nc")

    print("[LOG] Running experiment: Wochentag = Saturday")
    run_experiment(overrides={"Wochentag": "Saturday"}, outfile="predictions_wochentag_saturday.nc")

    print("[LOG] Running experiment: Wochentag = Sunday")
    run_experiment(overrides={"Wochentag": "Sunday"}, outfile="predictions_wochentag_sunday.nc")

    print("[LOG] Running experiment: Wochentag randomly from Monday..Friday")
    run_experiment(overrides={"Wochentag": "weekday_random"}, outfile="predictions_wochentag_weekdayrandom.nc")

    for hr in [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]:
        print(f"[LOG] Running experiment: Uhrzeit = {hr}")
        run_experiment(
            overrides={"Uhrzeit": hr},
            outfile=f"predictions_uhrzeit_{int(hr)}.nc"
        )

    print("[LOG] Running combined experiment with snow_cover=0.8, temperature_celsius=0.0")
    run_experiment(
        overrides={"snow_cover": 0.8, "temperature_celsius": 0.0},
        outfile="predictions_snowcover0.8_temp0.0.nc"
    )
