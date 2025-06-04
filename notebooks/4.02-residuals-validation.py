#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.base import clone
import numpy as np


print("[LOG] Loading trained pipeline model...")
# Load the entire pipeline (which includes the preprocessor and regressor)
pipeline_model = joblib.load("../data/interim/best_RF-Eintreffzeit.pkl")
pipeline_model.named_steps['rf'].n_jobs = 30

print("[LOG] Loading original training data...")
train_data = pd.read_parquet("../data/interim/selected-data_2024.parquet")

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

X = train_data[feature_columns]
y = train_data["Eintreffzeit"]

y_pred = pipeline_model.predict(X)
residuals = y - y_pred

y_pred

# ------------------------------
# Residuals vs. Fitted Values Plot
# ------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(y, residuals, alpha=0.5)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Tatsächliche Hilfsfrist Hilfsfrist [Minuten]")
plt.ylabel("Residuum")
plt.title("Residuen in Abhängigkeit von den Vorhersagen")
plt.xlim(4, 18)
plt.ylim(-15, 15)
plt.tight_layout()
plot_output_path = "../reports/figures/4.02-Residuen-Vorhersage.png"
plt.savefig(plot_output_path, dpi=300)
plt.show()

x_vals = np.unique(y)

# collect residuals for each x
data = [residuals[y == xv] for xv in x_vals]

plt.figure(figsize=(6,4))
plt.boxplot(data, positions=x_vals, widths=0.6, showfliers=False)
plt.axhline(0, color='black', linestyle='--')

plt.xticks(
    x_vals,                           # positions
    [str(int(x)) for x in x_vals]    # labels as integers
)

plt.xlabel("Tatsächliche Hilfsfrist [Minuten]")
plt.ylabel("Residuum")
plt.title("Residuen in Abhängigkeit von der Hilfsfrist")
plt.xlim(x_vals.min() - 1, x_vals.max() + 1)
plt.ylim(-15,15)
plt.tight_layout()
plt.savefig("../reports/figures/4.02-Residuen-Boxplot.png", dpi=300)
plt.show()

# ------------------------------
# Histogram of Residuals
# ------------------------------
plt.figure(figsize=(6, 4))
bins = np.arange(-15.5, 16.0, 1.0)
plt.hist(residuals, bins=bins, edgecolor="black", alpha=0.7)
plt.xlabel("Residuum [Minuten]")
plt.ylabel("Häufigkeit")
plt.title("Verteilung der Residuen")
plt.xlim(-15, 15)
plt.ylim(0, 5000)
plt.tight_layout()
plot_output_path = "../reports/figures/4.02-Residuen-Verteilung.png"
plt.savefig(plot_output_path, dpi=300)
plt.show()

# ------------------------------
# Q-Q Plot for Normality of Residuals
# ------------------------------
fig, ax = plt.subplots(figsize=(6, 4))

sm.qqplot(residuals, line="45", fit=True, ax=ax)

# now set labels/titles on that same axes
ax.set_title("Q-Q-Diagramm der Residuen")
ax.set_xlabel("Theoretische Quantile (Normalverteilung)")
ax.set_ylabel("Beobachtete Quantile der Residuen")

fig.tight_layout()
fig.savefig("../reports/figures/4.02-Residuen-QQ.png", dpi=300)
plt.show()

residuals.describe()

(residuals < 0).mean()

from sklearn.metrics import mean_squared_error, r2_score
import math

rmse_sklearn = math.sqrt(mean_squared_error(y, y_pred))
print(f"RMSE: {rmse_sklearn:.3f}")

# R²
r2_sklearn = r2_score(y, y_pred)
print(f"R²: {r2_sklearn:.3f}")
