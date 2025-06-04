#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import joblib
os.nice(20)

data = pd.read_parquet("../data/interim/selected-data.parquet")
print("\n=== Merged Dataset ===")
print("Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nMerged DataFrame info:")
print(data.info())

# Separate features and target
X = data[["Standort FZ",
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
           "hourly_snowfall"]]
y = data["Fahrzeit"]

# One-Hot Encode categorical variables
categorical_features = X.select_dtypes(include=['object', 'category']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Leave all other columns (e.g., numerical) as-is
)
# X = pd.get_dummies(X)

from sklearn.model_selection import RandomizedSearchCV

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(
        n_jobs=5,
        random_state=42
    ))
])

# Define parameter grid
param_dist = {
    'rf__n_estimators': [100,200,300,400,500],
    'rf__max_depth': [20, 30, None],
    'rf__min_samples_split': [2, 10, 15, 20],
    'rf__min_samples_leaf': [1, 2, 4, 6, 8],
    'rf__bootstrap': [True, False],
    'rf__max_features': [None, "sqrt"]
}

# Define scoring metrics
scoring = {
    'R2': 'r2',
    'RMSE': 'neg_root_mean_squared_error'
}

# Perform Randomized Search
rf_random = RandomizedSearchCV(
    estimator=pipeline,  # Directly optimize RandomForestRegressor, NOT another RandomizedSearchCV
    param_distributions=param_dist,
    n_iter=200,  # Number of different combinations to try
    cv=5,  # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=6,
    scoring=scoring,
    refit='RMSE',  # Refit using RMSE
    pre_dispatch="2*n_jobs"
)

# Fit the model
with joblib.parallel_backend('threading'):  
    rf_random.fit(X, y)
joblib.dump(rf_random.best_estimator_, "../data/interim/best_RF-Fahrzeit.pkl")
pd.DataFrame(rf_random.cv_results_).to_parquet("../data/interim/cv_results_Fahrzeit.parquet", index=False)

# Best parameters
print("Best Hyperparameters:", rf_random.best_params_)

import pandas as pd

# Convert search results to a DataFrame
cv_results = pd.DataFrame(rf_random.cv_results_)
cv_results

cv_results.sort_values("rank_test_RMSE")

# In[ ]:

