#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
os.nice(20)

data = pd.read_parquet("../data/interim/selected-data.parquet")
data.info()

# ------------------------------------------------------------------------------
# 2) Define target and features for each model
# ------------------------------------------------------------------------------
# Our target is always the actual arrival time:
y = data["Eintreffzeit"]

# Model 1: Directly use ORS-anfahrtszeit ("duration") as the prediction
#          -> No training needed, but for consistent code, we'll treat it
#             like a feature set. The final prediction is just X1_test itself.
X1 = data[["duration"]]

# Model 2: Linear regression using only "duration" as a predictor
X2 = data[["duration"]]

# Model 3: Random Forest using "Standort FZ", "EINSATZORT_lat", "EINSATZORT_lon",
#          "Wochentag", "Monat", "Uhrzeit"
X3 = data[["Standort FZ",
           "EINSATZORT_lat",
           "EINSATZORT_lon",
           "Wochentag",
           "Monat",
           "Uhrzeit"]]
X3 = pd.get_dummies(X3, drop_first=True)

# Model 4: Random Forest with extra data:
#   "Standort FZ","EINSATZORT_lat","EINSATZORT_lon","Wochentag","Monat","Uhrzeit",
#   "distance", "duration", "temperature_celsius", "snow_cover",
#   "dewpoint_temperature", "hourly_precipitation", "hourly_snowfall"
X4 = data[["Standort FZ",
           "EINSATZORT_lat",
           "EINSATZORT_lon",
           "Wochentag",
           "Monat",
           "Feiertag",
           "Ferien",
           "Uhrzeit",
           "distance",
           "duration",
           "temperature_celsius",
           "snow_cover",
           "dewpoint_temperature",
           "hourly_precipitation",
           "hourly_snowfall"]]
X4 = pd.get_dummies(X4, drop_first=True)

# ------------------------------------------------------------------------------
# 3) Set up bootstrap parameters & storage arrays
# ------------------------------------------------------------------------------
n_iterations = 100
# 80% of the data to sample (then we do a 70/30 train–test split on that sample)
n_samples = int(len(data) * 0.8)

# Storage lists for mean-squared errors of each model
mse_model1 = []
mse_model2 = []
mse_model3 = []
mse_model4 = []

# -----------------------------
# 4) Main loop: Use the same 80/20 train-test split for all models
# -----------------------------
for i in range(n_iterations):
    # Use a different random_state for each iteration so that the split changes every time.
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.2, random_state=i)
    
    # Extract the same training and test sets for all models.
    X1_train, X1_test = X1.iloc[train_idx], X1.iloc[test_idx]
    X2_train, X2_test = X2.iloc[train_idx], X2.iloc[test_idx]
    X3_train, X3_test = X3.iloc[train_idx], X3.iloc[test_idx]
    X4_train, X4_test = X4.iloc[train_idx], X4.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # -- Model 1: Direct usage of duration (no training) --
    y_pred1 = X1_test["duration"].values  
    mse1 = mean_squared_error(y_test, y_pred1)
    mse_model1.append(mse1)
    
    # -- Model 2: Linear Regression (duration as predictor) --
    lr = LinearRegression()
    lr.fit(X2_train, y_train)
    y_pred2 = lr.predict(X2_test)
    mse2 = mean_squared_error(y_test, y_pred2)
    mse_model2.append(mse2)
    
    # -- Model 3: Random Forest with limited features --
    rf3 = RandomForestRegressor(n_jobs=-1, random_state=42)
    rf3.fit(X3_train, y_train)
    y_pred3 = rf3.predict(X3_test)
    mse3 = mean_squared_error(y_test, y_pred3)
    mse_model3.append(mse3)
    
    # -- Model 4: Random Forest with extra features --
    rf4 = RandomForestRegressor(n_jobs=-1, random_state=42)
    rf4.fit(X4_train, y_train)
    y_pred4 = rf4.predict(X4_test)
    mse4 = mean_squared_error(y_test, y_pred4)
    mse_model4.append(mse4)
    
    print("{:} mse1: {:.4f}, mse2: {:.4f}, mse3: {:.4f}, mse4: {:.4f}".format(i, mse1, mse2, mse3, mse4))

# ------------------------------------------------------------------------------
# 5) Convert MSE lists to numpy arrays for easy post-processing
# ------------------------------------------------------------------------------
rmse_model1 = np.sqrt( np.array(mse_model1))
rmse_model2 = np.sqrt( np.array(mse_model2))
rmse_model3 = np.sqrt( np.array(mse_model3))
rmse_model4 = np.sqrt( np.array(mse_model4))

# ------------------------------------------------------------------------------
# 6) Example: Print out average MSE for each model
# ------------------------------------------------------------------------------
print("\nAverage RMSE across bootstraps:")
print(f"Model 1 (Direct ORS duration): {rmse_model1.mean():.4f}")
print(f"Model 2 (Linear on duration) : {rmse_model2.mean():.4f}")
print(f"Model 3 (RF, base features)  : {rmse_model3.mean():.4f}")
print(f"Model 4 (RF, extra features) : {rmse_model4.mean():.4f}")

# Convert differences to numpy array for analysis
rmse_diffs = np.array(rmse_model1-rmse_model2)

# Calculate summary statistics
mean_diff = np.mean(rmse_diffs)
ci_lower = np.percentile(rmse_diffs, 2.5)
ci_upper = np.percentile(rmse_diffs, 97.5)

print("Mean difference in MSE (rmse1 - rmse2): {:.4f}".format(mean_diff))
print("95% bootstrap CI: [{:.4f}, {:.4f}]".format(ci_lower, ci_upper))

# check what proportion of bootstraps show a positive difference
p_value = np.mean(rmse_diffs <= 0)
print("Proportion of bootstraps where full model did NOT improve performance: {:.4f}".format(p_value))

# Convert differences to numpy array for analysis
rmse_diffs = np.array(rmse_model2-rmse_model3)

# Calculate summary statistics
mean_diff = np.mean(rmse_diffs)
ci_lower = np.percentile(rmse_diffs, 2.5)
ci_upper = np.percentile(rmse_diffs, 97.5)

print("Mean difference in MSE (rmse1 - rmse2): {:.4f}".format(mean_diff))
print("95% bootstrap CI: [{:.4f}, {:.4f}]".format(ci_lower, ci_upper))

# check what proportion of bootstraps show a positive difference
p_value = np.mean(rmse_diffs <= 0)
print("Proportion of bootstraps where full model did NOT improve performance: {:.4f}".format(p_value))

# Convert differences to numpy array for analysis
rmse_diffs = np.array(rmse_model3-rmse_model4)

# Calculate summary statistics
mean_diff = np.mean(rmse_diffs)
ci_lower = np.percentile(rmse_diffs, 2.5)
ci_upper = np.percentile(rmse_diffs, 97.5)

print("Mean difference in MSE (rmse3 - rmse4): {:.4f}".format(mean_diff))
print("95% bootstrap CI: [{:.4f}, {:.4f}]".format(ci_lower, ci_upper))

# check what proportion of bootstraps show a positive difference
p_value = np.mean(rmse_diffs <= 0)
print("Proportion of bootstraps where full model did NOT improve performance: {:.4f}".format(p_value))

import matplotlib.pyplot as plt

# Calculate differences and error bars for each model comparison

# Model 2 - Model 1
diff_12 = rmse_model1 - rmse_model2
mean_diff_12 = diff_12.mean()
ci_lower_12 = np.percentile(diff_12, 2.5)
ci_upper_12 = np.percentile(diff_12, 97.5)
error_12 = [[mean_diff_12 - ci_lower_12], [ci_upper_12 - mean_diff_12]]

# Model 3 - Model 2
diff_23 = rmse_model2 - rmse_model3
mean_diff_23 = diff_23.mean()
ci_lower_23 = np.percentile(diff_23, 2.5)
ci_upper_23 = np.percentile(diff_23, 97.5)
error_23 = [[mean_diff_23 - ci_lower_23], [ci_upper_23 - mean_diff_23]]

# Model 4 - Model 3
diff_34 = rmse_model3 - rmse_model4
mean_diff_34 = diff_34.mean()
ci_lower_34 = np.percentile(diff_34, 2.5)
ci_upper_34 = np.percentile(diff_34, 97.5)
error_34 = [[mean_diff_34 - ci_lower_34], [ci_upper_34 - mean_diff_34]]

comparisons = ["Modell 1 - Modell 2", "Modell 2 - Modell 3", "Modell 3 - Modell 4"]
mean_diffs = [mean_diff_12, mean_diff_23, mean_diff_34]

# Combine the asymmetric error bars into one structure for matplotlib
errors = [ [error_12[0][0], error_23[0][0], error_34[0][0]], 
           [error_12[1][0], error_23[1][0], error_34[1][0]] ]

fig, ax = plt.subplots(figsize=(6,4))
bars = plt.bar(["Modell 1","Modell 2","Modell 3","Modell 4"], [rmse_model1.mean(),rmse_model2.mean(),rmse_model3.mean(),rmse_model4.mean()], yerr=[rmse_model1.std(),rmse_model2.std(),rmse_model3.std(),rmse_model4.std()], capsize=10, color='skyblue')
plt.ylabel('RMSE [min]')
plt.title('Generalisierungsfehler der Modelle')

# Function to draw significance markers
def add_significance(ax, x1, x2, y, h, text):
    """
    Draw a significance marker between x1 and x2 at height y with a small vertical offset h,
    and place the significance text above the line.
    """
    # Draw the horizontal line
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    # Place the text (star)
    ax.text((x1+x2)*0.5, y+h, text, ha='center', va='bottom', color='k')

# Get the x-coordinates for each bar (centers)
x_coords = [bar.get_x() + bar.get_width()/2 for bar in bars]
# Get the bar heights
heights = [bar.get_height() for bar in bars]
# Define an offset for the significance line
offset = max(heights) * 0.05

# Define a significance threshold
alpha = 0.05

y = max(heights[0], heights[1]) + offset
add_significance(ax, x_coords[0], x_coords[1], y, offset*0.5, '*')

y = max(heights[1], heights[2]) + offset*2  
add_significance(ax, x_coords[1], x_coords[2], y, offset*0.5, '*')

y = max(heights[2], heights[3]) + offset  
add_significance(ax, x_coords[2], x_coords[3], y, offset*0.5, '*')

plt.savefig("../reports/figures/3.02-Generalisierungsfehler-Modelle.png", dpi=300)
plt.show()

# Calculate the mean and std for each model
means = [rmse_model1.mean(), rmse_model2.mean(), rmse_model3.mean(), rmse_model4.mean()]
stds = [rmse_model1.std(), rmse_model2.std(), rmse_model3.std(), rmse_model4.std()]

# Format the values to a string with two decimal places and a plus-minus symbol
formatted = [f"{m:.2f} ± {s:.2f}" for m, s in zip(means, stds)]

# Create the DataFrame
models = ["Modell 1", "Modell 2", "Modell 3", "Modell 4"]
df = pd.DataFrame({"Modell": models, "Mittlerer RMSE": formatted})

print(df)

from scipy.stats import ttest_rel

statistic, p_value = ttest_rel(mse_model1, mse_model2)  # Welch’s t-test
print("t-test p-value:", p_value)

statistic, p_value = ttest_rel(mse_model2, mse_model3)  # Welch’s t-test
print("t-test p-value:", p_value)

statistic, p_value = ttest_rel(mse_model3, mse_model4)  # Welch’s t-test
print("t-test p-value:", p_value)

# In[ ]:

