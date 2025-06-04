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
model = joblib.load("../data/interim/best_RF-Eintreffzeit.pkl")

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
base_predictions = model.predict(X_sample)

results = pd.DataFrame({"base": base_predictions})

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
    results[name] = model.predict(X_mod)

# ----------------------------------------------------------------------------------------
# Run experiments by modifying "Uhrzeit" at finer resolution
# ----------------------------------------------------------------------------------------
time_steps = np.arange(0, 24, 0.1)

# Run Uhrzeit experiments at finer resolution
for time in time_steps:
    run_experiment({"Uhrzeit": time}, f"Uhrzeit_{time:.1f}")

# Convert results to DataFrame
results = pd.DataFrame(results)

import matplotlib.pyplot as plt
from scipy import stats
# ----------------------------------------------------------------------------------------
# Compute statistical metrics
# ----------------------------------------------------------------------------------------
differences = results.drop(columns=["base"]).sub(results["base"], axis=0)

diff_stats = differences.agg(["mean", "std"], axis=0)

# Combine into a DataFrame
stats_df = pd.DataFrame({
    "mean_diff": diff_stats.loc["mean"],
    "std_diff": diff_stats.loc["std"]
})

# ----------------------------------------------------------------------------------------
# Plot mean difference across different "Uhrzeit" values
# ----------------------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(time_steps, stats_df["mean_diff"], label="Mean Difference", color="blue")

plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel("Uhrzeit [Stunden])")
plt.ylabel("Differenz [Min]")
plt.title("Einfluss der Uhrzeit auf die Hilfsfrist")
plt.tight_layout()
plot_output_path = "../reports/figures/2.08-Uhrzeit-Hilfsfrist.png"
plt.savefig(plot_output_path, dpi=300)
plt.show()
plt.show()

# In[ ]:

