#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

cv_results = pd.read_parquet("../data/interim/cv_results_Eintreffzeit.parquet")

# RMSE is stored as a negative value (to maximize score); convert it to positive RMSE
cv_results["RMSE"] = -cv_results["mean_test_RMSE"]

cv_results

hyperparameters = [
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "bootstrap",
    "max_features"
]

# Loop over each hyperparameter to create a boxplot and a summary table
for param in hyperparameters:
    col_str = "param_rf__" + param
    cv_results[col_str] = cv_results[col_str].astype(str)

    # 1) Group by hyperparameter value
    groups = cv_results.groupby(col_str)["RMSE"]

    # 2) Build and sort the summary table
    summary = groups.agg(["mean", "std"]).reset_index()
    summary = summary.rename(
        columns={col_str: param, "mean": "Mean_RMSE", "std": "Std_RMSE"}
    )
    # Try numeric conversion for correct ascending order
    summary[param] = pd.to_numeric(summary[param], errors="ignore")
    summary = summary.sort_values(by=param).reset_index(drop=True)

    # 3) Extract the sorted labels (as strings) for consistent boxplot ordering
    sorted_labels = summary[param].astype(str).tolist()
    sorted_data   = [groups.get_group(lbl).values for lbl in sorted_labels]

    # --- Boxplot with sorted labels ---
    plt.figure(figsize=(6, 4))
    plt.boxplot(sorted_data, labels=sorted_labels)
    plt.xlabel(param)
    plt.ylabel("RMSE [min]")
    plt.title(f"RMSE Verteilung für Hyperparameter {param}")
    plt.tight_layout()
    plt.savefig(f"../reports/figures/1.16_boxplot_{param}.png")
    plt.close()

    # --- (Optional) print the summary to console ---
    print(f"\nSummary table for {param}:")
    summary[["Mean_RMSE", "Std_RMSE"]] = summary[["Mean_RMSE", "Std_RMSE"]].round(2)
    print(summary.to_string(index=False))

print("\nAnalysis complete. Boxplots and summary tables have been saved.")

cv_results.sort_values("rank_test_RMSE")

cv_results.sort_values("rank_test_RMSE")

print(cv_results[["rank_test_RMSE","RMSE","mean_test_R2"]].sort_values(by=  "rank_test_RMSE") )
