#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_parquet("../data/interim/test_dataset.parquet")
print("\n=== Merged Dataset ===")
print("Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nMerged DataFrame info:")
print(data.info())

# Define predictions and ground truth
y_true = data["Anfahrtszeit"]
y_pred = data["duration"]

# Compute metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)

# Display results
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Plot actual vs. predicted values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color='red')  # Ideal prediction line
plt.xlabel("Actual Anfahrtszeit")
plt.ylabel("Predicted Duration")
plt.title("Actual vs. Predicted")

# Plot residuals
plt.subplot(1, 2, 2)
sns.histplot(y_true - y_pred, bins=30, kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")

plt.tight_layout()
plt.show()

# In[ ]:

