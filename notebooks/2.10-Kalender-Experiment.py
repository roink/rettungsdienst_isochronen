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

def run_experiment(overrides, name):
    modified = sample_df.copy()
    if overrides is not None:
        # special handling of random weekday baseline
        val = overrides.get("Wochentag", None)
        if val == "weekday_random":
            print(f"[LOG] Randomly assigning Montag–Freitag to Wochentag for '{name}'...")
            modified["Wochentag"] = np.random.choice(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                size=len(modified),
            )
        else:
            for col, v in overrides.items():
                modified[col] = v

    X_mod = modified.drop(columns=["Fahrzeit", "Eintreffzeit"], errors="ignore")
    results[name] = model.predict(X_mod)

cat_experiments = {
    "Feiertag": [False, True],
    "Ferien":   [False, True],
    # compare Samstag and Sonntag each against a random Mon–Fri draw
    "Wochentag": ["weekday_random", "Saturday", "Sunday"]
}

print("[LOG] Running categorical experiments...")
for var, values in cat_experiments.items():
    for val in values:
        run_experiment({var: val}, f"{var}_{val}")

# ----------------------------------------------------------------------------------------
# Compute differences & statistics
# ----------------------------------------------------------------------------------------
diffs = results.drop(columns=["base"]).sub(results["base"], axis=0)
agg = diffs.agg(["mean", "std"], axis=0)
pvals = diffs.apply(lambda col: stats.ttest_1samp(col, 0).pvalue)
dvals = agg.loc["mean"].abs() / agg.loc["std"]
sem   = agg.loc["std"] / np.sqrt(n_samples)

stats_df = pd.DataFrame({
    "mean_diff": agg.loc["mean"],
    "std_diff":  agg.loc["std"],
    "cohens_d":  dvals,
    "p_value":   pvals,
    "sem":       sem
})

# show table
print(stats_df)

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

import matplotlib.pyplot as plt

# === 1. Feiertag effect ===
# select the two rows
feiertag_idx = ["Feiertag_False", "Feiertag_True"]
feiertag_stats = stats_df.loc[feiertag_idx]

plt.figure(figsize=(6,4))
plt.bar(
    ["Kein Feiertag", "Feiertag"],
    feiertag_stats["mean_diff"],
    yerr=feiertag_stats["sem"],
    capsize=5
)
plt.axhline(0, linestyle="--", linewidth=0.8)
plt.ylabel("Δ Hilfsfrist [min]")
plt.title("Effekt von Feiertagen")
plt.tight_layout()
plot_output_path = "../reports/figures/2.10-Feiertage.png"
plt.savefig(plot_output_path, dpi=300)
plt.show()

# === 2. Ferien effect ===
ferien_idx   = ["Ferien_False", "Ferien_True"]
ferien_stats = stats_df.loc[ferien_idx]

plt.figure(figsize=(6,4))
plt.bar(
    ["Keine Ferien", "Ferien"],
    ferien_stats["mean_diff"],
    yerr=ferien_stats["sem"],
    capsize=5
)
plt.axhline(0, linestyle="--", linewidth=0.8)
plt.ylabel("Δ Hilfsfrist [min]")
plt.title("Effekt von Ferien")
plt.tight_layout()
plot_output_path = "../reports/figures/2.10-Ferien.png"
plt.savefig(plot_output_path, dpi=300)
plt.show()

# === 3. Wochentag effect ===
wochentag_idx   = ["Wochentag_weekday_random", "Wochentag_Saturday", "Wochentag_Sunday"]
wochentag_stats = stats_df.loc[wochentag_idx]
labels = ["Werktag\n(Mo–Fr)", "Samstag", "Sonntag"]

plt.figure(figsize=(6,4))
plt.bar(
    labels,
    wochentag_stats["mean_diff"],
    yerr=wochentag_stats["sem"],
    capsize=5
)
plt.axhline(0, linestyle="--", linewidth=0.8)
plt.ylabel("Δ Hilfsfrist [min]")
plt.title("Effekt von Wochentagen")
plt.tight_layout()
plot_output_path = "../reports/figures/2.10-Wochentage.png"
plt.savefig(plot_output_path, dpi=300)
plt.show()

row_mapping = {
    'Feiertag_False': 'Kein Feiertag',
    'Feiertag_True':  'Feiertag',
    'Ferien_False':   'Keine Ferien',
    'Ferien_True':    'Ferien',
    'Wochentag_weekday_random': 'Werktag (Mo–Fr)',
    'Wochentag_Saturday': 'Samstag',
    'Wochentag_Sunday':   'Sonntag'
}
col_mapping = {
    'mean_diff': 'Δ Hilfsfrist',
    'std_diff':  'STD',
    'p_value':   'p-Wert'
}

# Apply renaming and select only the desired columns, rounding to 2 decimals
variant_df = (
    stats_df
    .rename(index=row_mapping)
    .loc[:, ['mean_diff', 'std_diff', 'p_value']]
    .rename(columns=col_mapping)
)
variant_df[['Δ Hilfsfrist', 'STD']] = variant_df[['Δ Hilfsfrist', 'STD']] * 60
variant_df = variant_df.round(2)

variant_df

import matplotlib.pyplot as plt

# Define labels and corresponding index keys
labels = [
    "Kein\nFeiertag",
    "Feiertag",
    "Keine\nFerien",
    "Ferien",
    "Werktag\n(Mo–Fr)",
    "Samstag",
    "Sonntag"
]
idx = [
    "Feiertag_False",
    "Feiertag_True",
    "Ferien_False",
    "Ferien_True",
    "Wochentag_weekday_random",
    "Wochentag_Saturday",
    "Wochentag_Sunday"
]

means = 60*stats_df.loc[idx, "mean_diff"]
sems = 60*stats_df.loc[idx, "std_diff"]

plt.figure(figsize=(6, 4))
plt.bar(labels, means, yerr=sems, capsize=5)
plt.axhline(0, linestyle="--", linewidth=0.8)
plt.ylabel("Δ Hilfsfrist [Sek]")
plt.title("Kalendarische Effekte")
plt.xticks( )
plt.tight_layout()
plot_output_path = "../reports/figures/2.10-Kalendar.png"
plt.savefig(plot_output_path, dpi=300)
plt.show()

# In[ ]:

