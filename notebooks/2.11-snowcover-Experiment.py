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
model = joblib.load("../data/interim/best_RF-Eintreffzeit.pkl")

print("[LOG] Loading training data...")
train_data = pd.read_parquet("../data/interim/selected-data.parquet")

train_data.info()

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
# Run experiments by modifying "Uhrzeit" at finer resolution
# ----------------------------------------------------------------------------------------

def run_experiment(overrides, name):
    modified_sample = sample_df.copy()
    for col, val in overrides.items():
        modified_sample[col] = val

    X_mod = modified_sample.drop(columns=["Fahrzeit", "Eintreffzeit"], errors="ignore")

    results[name] = model.predict(X_mod)

snow_covers = np.arange(0, 1, 0.01)
    
# Run temperature experiments at finer resolution
for snow_cover in snow_covers:
    run_experiment({"snow_cover": snow_cover}, f"snow_cover_{snow_cover:.2f}")

# Convert results to DataFrame
results = pd.DataFrame(results)

# ----------------------------------------------------------------------------------------
# Compute statistical metrics
# ----------------------------------------------------------------------------------------
differences = results.drop(columns=["base"]).sub(results["base"], axis=0)

diff_stats = differences.agg(["mean", "std"], axis=0)
p_values = differences.apply(lambda col: stats.ttest_1samp(col, 0).pvalue)
effect_sizes = np.abs(diff_stats.loc["mean"] / diff_stats.loc["std"])
sem = diff_stats.loc["std"] / np.sqrt(n_samples)

# Combine into a DataFrame
stats_df = pd.DataFrame({
    "mean_diff": diff_stats.loc["mean"],
    "std_diff": diff_stats.loc["std"],
    "cohens_d": effect_sizes,
    "p_value": p_values,
    "sem" : sem
})

# Define interpretation functions
def interpret_cohens_d(d):
    if d < 0.2:
        return "Negligible effect"
    elif d < 0.5:
        return "Small effect"
    elif d < 0.8:
        return "Medium effect"
    else:
        return "Large effect"

def interpret_p_value(p):
    if p > 0.05:
        return "Not statistically significant"
    elif p < 0.001:
        return "Strongly statistically significant"
    else:
        return "Statistically significant"

# Apply interpretations
stats_df["cohens_d_interpretation"] = stats_df["cohens_d"].apply(interpret_cohens_d)
stats_df["p_value_interpretation"] = stats_df["p_value"].apply(interpret_p_value)

# Display the DataFrame
stats_df

plt.figure(figsize=(6, 4))
plt.plot(100*snow_covers, stats_df["mean_diff"], label="Differenz [min]", color="blue")
plt.fill_between(100*snow_covers, 
                 stats_df["mean_diff"] - (stats_df["std_diff"] ), 
                 stats_df["mean_diff"] + (stats_df["std_diff"] ), 
                 color="blue", alpha=0.2, label="")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel("Schneebedeckung [%]")
plt.ylabel("Δ Hilfsfrist [min]")
plt.title("Effekt der Schneebedeckung auf die Hilfsfrist")
plt.tight_layout()
plot_output_path = "../reports/figures/2.11-Snowcover-Hilfsfrist.png"
plt.savefig(plot_output_path, dpi=300)
plt.show()

