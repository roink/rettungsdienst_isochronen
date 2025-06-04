#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
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
y = data["Eintreffzeit"] <= 8

# One-Hot Encode categorical variables
categorical_features = X.select_dtypes(include=['object', 'category']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Leave all other columns (e.g., numerical) as-is
)
# X = pd.get_dummies(X)

from sklearn.model_selection import GridSearchCV
from scipy.stats import randint

from scipy.stats import randint

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestClassifier(
        n_jobs=30,
        random_state=42
    ))
])

# Define parameter grid
param_dist = {
    'rf__n_estimators': [300,400,500,600],
    'rf__max_depth': [15, 20],
    'rf__min_samples_split': [15, 20],
    'rf__min_samples_leaf': [ 6, 8],
    'rf__bootstrap': [True],
    'rf__max_features': ["sqrt"]
}

scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1'
}

rf_random = GridSearchCV(
    estimator=pipeline,
    param_grid=param_dist,
    scoring=scoring,      # Pass the dictionary of scoring metrics
    refit='f1',           # Use the F1 score for selecting the best model
    cv=5,
    verbose=2,
    n_jobs=2,
    pre_dispatch="2*n_jobs"
)

# Fit the model
with joblib.parallel_backend('threading'):  # Alternative: 'multiprocessing'
    rf_random.fit(X, y)
joblib.dump(rf_random.best_estimator_, "../data/interim/best_RF-Eintreffzeit-Classifier8min.pkl")
joblib.dump(rf_random, "../data/interim/rf_random_Eintreffzeit-Classifier8min.pkl")

# Best parameters
print("Best Hyperparameters:", rf_random.best_params_)

import pandas as pd

# Convert search results to a DataFrame
cv_results = pd.DataFrame(rf_random.cv_results_)
cv_results

cv_results.sort_values("rank_test_f1")

cv_results[["rank_test_f1","mean_test_f1","mean_test_accuracy"]].sort_values("rank_test_f1")

# In[ ]:

