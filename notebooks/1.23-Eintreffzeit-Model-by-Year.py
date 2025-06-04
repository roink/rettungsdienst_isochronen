#!/usr/bin/env python
# coding: utf-8

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

os.nice(20)

# Load the best tuned model
best_model_path = "../data/interim/best_RF-Eintreffzeit.pkl"
best_model = joblib.load(best_model_path)
print("Loaded best model from:", best_model_path)

# Load the full dataset
data = pd.read_parquet("../data/interim/selected-data.parquet")
print("\n=== Full Dataset ===")
print("Shape:", data.shape)

# Check that the 'Jahr' column exists
if "Jahr" not in data.columns:
    raise ValueError("Column 'Jahr' not found in the dataset.")

# Get the unique years
years = sorted(data["Jahr"].unique())
print("Training separate models for years:", years)

# ---------------------------------------------------------
# Define feature columns and the target column for training
# ---------------------------------------------------------
feature_cols = [
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
target_col = "Eintreffzeit"

# Dictionary to store models for each year (if further use is required)
year_models = {}

# Process and train a model for each year
for year in years:
    print(f"\nProcessing data for year: {year}")
    
    # Filter the data for the specific year
    data_year = data[data["Jahr"] == year]
    print("Data shape for year", year, ":", data_year.shape)
    
    # Separate features (X) and target (y)
    X_year = data_year[feature_cols]
    y_year = data_year[target_col]
    
    # Clone the best_model pipeline to get a fresh pipeline with the same tuned hyperparameters
    pipeline_year = clone(best_model)
    pipeline_year.set_params(rf__n_jobs=30)
    
    # Fit the pipeline on the subset of the data
    pipeline_year.fit(X_year, y_year)
    
    # Evaluate training performance (optional)
    y_pred = pipeline_year.predict(X_year)
    r2 = r2_score(y_year, y_pred)
    mae = mean_absolute_error(y_year, y_pred)
    rmse = np.sqrt(mean_squared_error(y_year, y_pred))
    print(f"Year {year} training metrics: R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    # Save the trained model for this year
    model_filename = f"../data/interim/best_Eintreffzeit_{year}.pkl"
    joblib.dump(pipeline_year, model_filename)
    print(f"Saved model for year {year} at: {model_filename}")
    
    # Optionally, store the model in the dictionary
    year_models[year] = pipeline_year

print("\nTraining complete for all years.")

pipeline_year

pipeline_year.named_steps["rf"]

rf = pipeline_year.named_steps['rf']
rf.n_jobs = 20
