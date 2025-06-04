#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.stats import f
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
os.nice(20)

data = pd.read_parquet("../data/interim/selected-data.parquet")
data.info()

# ------------------------------------------------------------------------------
# 2) Define target and features for each model
# ------------------------------------------------------------------------------
# Our target is always the actual arrival time:
y = data["Eintreffzeit"]

# Model 4: Random Forest with extra data:
#   "Standort FZ","EINSATZORT_lat","EINSATZORT_lon","Wochentag","Monat","Uhrzeit",
#   "distance", "duration", "temperature_celsius", "snow_cover",
#   "dewpoint_temperature", "hourly_precipitation", "hourly_snowfall"
X_reduced = data[["Standort FZ",
           "EINSATZORT_lat",
           "EINSATZORT_lon",
           "Wochentag",
           "Monat",
           "Feiertag",
           "Ferien",
           "Uhrzeit",
           "distance",
           "duration",
           "snow_cover",
           "dewpoint_temperature",
           "hourly_precipitation",
           "hourly_snowfall"]]

# Model 4: Random Forest with extra data:
#   "Standort FZ","EINSATZORT_lat","EINSATZORT_lon","Wochentag","Monat","Uhrzeit",
#   "distance", "duration", "temperature_celsius", "snow_cover",
#   "dewpoint_temperature", "hourly_precipitation", "hourly_snowfall"
Xfull = data[["Standort FZ",
           "EINSATZORT_lat",
           "EINSATZORT_lon",
           "Wochentag",
           "Monat",
           "Feiertag",
           "Ferien",
           "Uhrzeit",
           "distance",
           "duration",
           "temperature_celsius",
           "snow_cover",
           "dewpoint_temperature",
           "hourly_precipitation",
           "hourly_snowfall"]]

categorical_features = Xfull.select_dtypes(include=['object', 'category']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Leave all other columns (e.g., numerical) as-is
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(
        n_jobs=5,
        random_state=42
    ))
])

pipeline2 = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(
        n_jobs=5,
        random_state=42
    ))
])

train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42)

# Extract the same training and test sets for all models.
X_reduced_train, X_reduced_test = X_reduced.iloc[train_idx], X_reduced.iloc[test_idx]
X_full_train, X_full_test = Xfull.iloc[train_idx], Xfull.iloc[test_idx]

y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# 1) Fit the reduced model
rf_reduced = pipeline
rf_reduced.fit(X_reduced_train, y_train)
y_pred_red = rf_reduced.predict(X_reduced_test)

# 2) Fit the full model
rf_full = pipeline2
rf_full.fit(X_full_train, y_train)
y_pred_full = rf_full.predict(X_full_test)

# 3) Compute RSS for each
rss_red  = np.sum((y_test - y_pred_red)**2)
rss_full = np.sum((y_test - y_pred_full)**2)

print("rss_red: " + str(rss_red) )
print("rss_full: " + str(rss_full) )

# 4) Degrees of freedom
n_test   = len(y_test)
print("n_test: " + str(n_test) )
p_reduced = X_reduced_train.shape[1]
print("p_reduced: " + str(p_reduced) )

p_full    = X_full_train.shape[1]
print("p_full: " + str(p_full) )
delta_p  = p_full - p_reduced
print("delta_p: " + str(delta_p) )

# 5) Estimate MSE_full (residual variance)
mse_full = rss_full / (n_test - p_full)  
print("mse_full: " + str(mse_full) )

# 6) F–statistic
F_stat = (rss_red - rss_full) / (delta_p * mse_full)
df1 = delta_p
df2 = n_test - p_full

# 7) p‐value
p_value = f.sf(F_stat, df1, df2)

print(f"Partial F = {F_stat:.3f} on ({df1}, {df2}) df -> p = {p_value:.4f}")
