#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------------------------------------------------------------------
# Load model and training data
# ----------------------------------------------------------------------------------------
print("[LOG] Loading trained model...")
model = joblib.load("../data/interim/best_logRF-Eintreffzeit.pkl")

import shap
print("[LOG] Loading training data...")
train_data = pd.read_parquet("../data/interim/selected-data.parquet")
train_data = train_data[["Standort FZ",
           "EINSATZORT_lat",
           "EINSATZORT_lon",
           "Wochentag",
           "Monat",
           "Uhrzeit",
           "distance",
           "duration",
           "temperature_celsius",
           "snow_cover",
           "dewpoint_temperature",
           "hourly_precipitation",
           "hourly_snowfall"]]

# Preprocess X manually
X_preprocessed = model.named_steps['preprocessor'].transform(train_data)
X_sample = pd.DataFrame(X_preprocessed).sample(100, random_state=42)

# Access the underlying RandomForest
rf = model.named_steps['rf'].regressor_

# Explain
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample)

# In[ ]:

