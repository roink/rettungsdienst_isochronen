#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

data = pd.read_parquet("../data/interim/train_dataset.parquet")
print("\n=== Merged Dataset ===")
print("Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nMerged DataFrame info:")
print(data.info())

# Calculate Alarmverzögerung
data["Alarmverzögerung"] = data["Eintreffzeit"] - data["Fahrzeit"]

# Summary statistics
data["Alarmverzögerung"].describe()

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(data["Alarmverzögerung"], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel("Alarmverzögerung (minutes)")
plt.ylabel("Frequency")
plt.title("Distribution of Alarmverzögerung")
plt.grid(True)
plt.show()

# Calculate additional percentiles
data["Alarmverzögerung"].quantile([0.90, 0.95, 0.97, 0.98, 0.99, 0.995])

# Boxplot to visualize outliers
plt.figure(figsize=(10, 6))
plt.boxplot(data["Alarmverzögerung"], vert=False, patch_artist=True)
plt.xlabel("Alarmverzögerung (minutes)")
plt.title("Boxplot of Alarmverzögerung")
plt.grid(True)
plt.show()

# Identify extreme outliers using the IQR method
Q1 = data["Alarmverzögerung"].quantile(0.25)
Q3 = data["Alarmverzögerung"].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = Q3 + 1.5 * IQR  # Upper bound for outliers

# Filter potential outliers
outliers = data[data["Alarmverzögerung"] > outlier_threshold]
outliers

len(outliers)/len(data)

outliers.describe()

non_outliers = data[data["Alarmverzögerung"] <= outlier_threshold]
# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(non_outliers["Alarmverzögerung"], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel("Alarmverzögerung (minutes)")
plt.ylabel("Frequency")
plt.title("Distribution of Alarmverzögerung")
plt.grid(True)
plt.show()


