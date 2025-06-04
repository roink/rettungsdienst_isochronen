#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import statsmodels.api as sm

print("[LOG] Loading trained pipeline model...")
# Load the entire pipeline (which includes the preprocessor and regressor)
pipeline_model = joblib.load("../models/best_logRF-Eintreffzeit.pkl")
pipeline_model.named_steps['rf'].n_jobs = 30

print("[LOG] Loading original training data...")
train_data = pd.read_parquet("../data/interim/selected-data_2024.parquet")

feature_columns = [
    "Standort FZ",
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
    "hourly_snowfall"
]

X = train_data[feature_columns]
y = train_data["Eintreffzeit"]

# ------------------------------
# Compute predictions and residuals
# ------------------------------
y_pred = pipeline_model.predict(X)
residuals = y - y_pred

# ------------------------------
# Residuals vs. Fitted Values Plot
# ------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Vorhergesagte Hilfsfrist [Minuten]")
plt.ylabel("Residuum")
plt.title("Residuen in Abhängigkeit von den Vorhersagen")
plt.tight_layout()
plot_output_path = "../reports/figures/4.02-Residuen-Vorhersage-2024.png"
plt.savefig(plot_output_path, dpi=300)
plt.show()

# ------------------------------
# Histogram of Residuals
# ------------------------------
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
plt.xlabel("Residuum [Minuten]")
plt.ylabel("Häufigkeit")
plt.title("Verteilung der Residuen")
plt.tight_layout()
plot_output_path = "../reports/figures/4.02-Residuen-Verteilung-2024.png"
plt.savefig(plot_output_path, dpi=300)
plt.show()

residuals.describe()

(residuals < 0).mean()

# In[ ]:

