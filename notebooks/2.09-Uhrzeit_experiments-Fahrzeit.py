#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import joblib

# ----------------------------------------------------------------------------------------
# Load model and training data
# ----------------------------------------------------------------------------------------
print("[LOG] Loading trained model...")
# Load the pipeline which already includes preprocessing
model_Fahrzeit = joblib.load("../data/interim/best_RF-Fahrzeit.pkl")
model_Eintreffzeit = joblib.load("../data/interim/best_RF-Eintreffzeit.pkl")

print("[LOG] Loading training data...")
# Use the same training data used in model training
train_data = pd.read_parquet("../data/interim/selected-data.parquet")

# ----------------------------------------------------------------------------------------
# Sample data for experiments
# ----------------------------------------------------------------------------------------
n_samples = 10000  # Define number of samples
seed = 42  # Fixed seed for reproducibility

print("[LOG] Sampling training data...")
sample_df = train_data.sample(n=n_samples, replace=True, random_state=seed).reset_index(drop=True)

# Prepare feature matrix for predictions (drop only the target column)
X_sample = sample_df.drop(columns=["Eintreffzeit"], errors="ignore")

# ----------------------------------------------------------------------------------------
# Run base experiment
# ----------------------------------------------------------------------------------------
print("[LOG] Running base experiment...")
base_predictions_Fahrzeit = model_Fahrzeit.predict(X_sample)
base_predictions_Eintreffzeit = model_Eintreffzeit.predict(X_sample)

results_Fahrzeit = pd.DataFrame({"base": base_predictions_Fahrzeit})
results_Hilfsfrist = pd.DataFrame({"base": base_predictions_Eintreffzeit})

# ----------------------------------------------------------------------------------------
# Run experiments by modifying single predictors
# ----------------------------------------------------------------------------------------
def run_experiment(overrides, name):
    modified_sample = sample_df.copy()
    for col, val in overrides.items():
        # Special handling for a randomized weekday
        if col == "Wochentag" and val == "weekday_random":
            modified_sample[col] = np.random.choice(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                size=n_samples,
            )
        else:
            modified_sample[col] = val

    # Drop the target column before prediction
    X_mod = modified_sample.drop(columns=["Eintreffzeit"], errors="ignore")
    results_Fahrzeit[name] = model_Fahrzeit.predict(X_mod)
    results_Hilfsfrist[name] = model_Eintreffzeit.predict(X_mod)

# ----------------------------------------------------------------------------------------
# Run experiments by modifying "Uhrzeit" at finer resolution
# ----------------------------------------------------------------------------------------
time_steps = np.arange(0, 24, 0.1)

# Run Uhrzeit experiments at finer resolution
for time in time_steps:
    run_experiment({"Uhrzeit": time}, f"Uhrzeit_{time:.1f}")

# Convert results to DataFrame
results_Fahrzeit = pd.DataFrame(results_Fahrzeit)
results_Hilfsfrist = pd.DataFrame(results_Hilfsfrist)

results_Fahrzeit

import matplotlib.pyplot as plt
from scipy import stats
# ----------------------------------------------------------------------------------------
# Compute statistical metrics
# ----------------------------------------------------------------------------------------
differences_Fahrzeit = results_Fahrzeit.drop(columns=["base"]).sub(results_Fahrzeit["base"], axis=0)

diff_stats_Fahrzeit = differences_Fahrzeit.agg(["mean", "std"], axis=0)

differences_Hilfsfrist = results_Hilfsfrist.drop(columns=["base"]).sub(results_Hilfsfrist["base"], axis=0)

diff_stats_Hilfsfrist = differences_Hilfsfrist.agg(["mean", "std"], axis=0)

# Combine into a DataFrame
stats_df = pd.DataFrame({
    "mean_diff_Fahrzeit": diff_stats_Fahrzeit.loc["mean"],
    "mean_diff_Hilfsfrist": diff_stats_Hilfsfrist.loc["mean"],
    "std_diff_Fahrzeit": diff_stats_Fahrzeit.loc["std"],
    "std_diff_Hilfsfrist": diff_stats_Hilfsfrist.loc["std"],
})

# ----------------------------------------------------------------------------------------
# Plot mean difference across different "Uhrzeit" values
# ----------------------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(time_steps, stats_df["mean_diff_Hilfsfrist"], label="Hilfsfrist")
plt.fill_between(time_steps, 
                 stats_df["mean_diff_Hilfsfrist"] - stats_df["std_diff_Hilfsfrist"], 
                 stats_df["mean_diff_Hilfsfrist"] + stats_df["std_diff_Hilfsfrist"], 
                 color="blue", alpha=0.2, label="")
plt.plot(time_steps, stats_df["mean_diff_Fahrzeit"], label="Anfahrtszeit")
plt.fill_between(time_steps, 
                 stats_df["mean_diff_Fahrzeit"] - stats_df["std_diff_Fahrzeit"], 
                 stats_df["mean_diff_Fahrzeit"] + stats_df["std_diff_Fahrzeit"], 
                 color="orange", alpha=0.2, label="")

plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel("Uhrzeit [h]")
plt.ylabel("Δ [min]")
plt.title("Effekt der Uhrzeit auf Anfahrtszeit und Hilfsfrist")
plt.legend()

plt.xlim([0,24])
plt.grid(True)
plt.tight_layout()
plot_output_path = "../reports/figures/2.09-Uhrzeit-Fahrzeit.png"
plt.savefig(plot_output_path, dpi=300)
plt.show()

stats_df.max()

# In[ ]:

