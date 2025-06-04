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
    print(f"[LOG] Running experiment: {name}")
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

# Define experiments (removed 'Feiertag' and 'Ferien' since these no longer exist in the new feature set)
experiments = {
    "Wochentag_Saturday": {"Wochentag": "Saturday"},
    "Wochentag_Sunday": {"Wochentag": "Sunday"},
    "Wochentag_WeekdayRandom": {"Wochentag": "weekday_random"},
}

# Run the defined experiments
for name, overrides in experiments.items():
    run_experiment(overrides, name)

# Run experiments for different Uhrzeit values
for hr in [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]:
    run_experiment({"Uhrzeit": hr}, f"Uhrzeit_{int(hr)}")

# Run a combined experiment for 'snow_cover' and 'temperature_celsius'
run_experiment({"snow_cover": 0.8, "temperature_celsius": 0.0}, "SnowCover0.8_Temp0.0")

results

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------
# Plot the experiment predictions across samples
# ----------------------------------------------------------------------------------------
results = results.sort_values(by="base").reset_index(drop=True)

plt.figure(figsize=(12, 6))
for col in results.columns:
    plt.plot(results.index, results[col], label=col, alpha=0.7)

plt.xlabel("Sample Index")
plt.ylabel("Predicted Value")
plt.title("Experiment Predictions Across Samples")
plt.legend(loc="upper right", fontsize="small", ncol=2)
plt.show()

from scipy import stats
# Compare each experiment’s predictions to the base prediction
differences = results.drop(columns=["base"]).sub(results["base"], axis=0)

# Compute mean and standard deviation of differences
diff_stats = differences.agg(["mean", "std"], axis=0)

# Compute p-values from t-tests against the null hypothesis that mean difference = 0
p_values = differences.apply(lambda col: stats.ttest_1samp(col, 0).pvalue)

# Compute Cohen's d (mean difference divided by standard deviation)
effect_sizes = np.abs(diff_stats.loc["mean"] / diff_stats.loc["std"])

# Combine all statistics into a single DataFrame
stats_df = pd.DataFrame({
    "mean_diff": diff_stats.loc["mean"],
    "std_diff": diff_stats.loc["std"],
    "cohens_d": effect_sizes,
    "p_value": p_values
})

# ----------------------------------------------------------------------------------------
# Plot mean differences with standard deviation error bars
# ----------------------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.bar(stats_df.index, stats_df["mean_diff"], yerr=stats_df["std_diff"], capsize=5, alpha=0.7)
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.xticks(rotation=90)
plt.ylabel("Mean Difference")
plt.title("Mean Differences and Standard Deviations for Experiments")
plt.show()

# ----------------------------------------------------------------------------------------
# Interpretation Functions for Statistical Analysis
# ----------------------------------------------------------------------------------------
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

print(stats_df)

# In[ ]:

