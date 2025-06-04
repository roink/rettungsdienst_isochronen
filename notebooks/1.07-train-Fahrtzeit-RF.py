#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_parquet("../data/interim/train_dataset.parquet")
print("\n=== Merged Dataset ===")
print("Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nMerged DataFrame info:")
print(data.info())

# Separate features and target
X = data.drop(columns=["Fahrzeit","Eintreffzeit",])
y = data["Fahrzeit"]

# One-Hot Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

y_pred.shape

# Define predictions and ground truth
y_true = y_test

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

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define parameter grid
param_dist = {
    'n_estimators': randint(50, 500),  # Number of trees
    'max_depth': [ 20, 30, None],  # Tree depth
    'min_samples_split': [15, 20, 10],  # Minimum samples to split
    'min_samples_leaf': [ 2, 4, 6, 8],  # Minimum samples per leaf
    'max_features': [ 'sqrt'],  # Features to consider per split
    'bootstrap': [True, False]  # Bootstrapping
}

# Randomized Search CV
rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=200,  # Number of different combinations to try
    cv=5,  # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1
)

rf_random.fit(X, y)

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
import numpy as np

# Get the best model
best_model = rf_random.best_estimator_

# Predict using cross-validation
y_pred = cross_val_predict(best_model, X, y, cv=5)

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"Cross-validation RMSE of the best model: {rmse:.4f}")

# Best parameters
print("Best Hyperparameters:", rf_random.best_params_)

import pandas as pd

# Convert search results to a DataFrame
cv_results = pd.DataFrame(rf_random.cv_results_)

# Select only relevant columns
cv_results = cv_results[["mean_test_score", "param_n_estimators", "param_max_depth", 
                         "param_min_samples_split", "param_min_samples_leaf", "param_max_features",
                         "param_bootstrap"]]

# Sort by best test score
cv_results = cv_results.sort_values(by="mean_test_score", ascending=False)

cv_results

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.scatterplot(x=cv_results["param_n_estimators"], y=cv_results["mean_test_score"])
plt.xlabel("Number of Estimators")
plt.ylabel("Mean Test Score (R²)")
plt.title("Effect of n_estimators on Model Performance")
plt.show()

# Convert None values to a string for plotting
cv_results["param_max_depth"] = cv_results["param_max_depth"].astype(str)

plt.figure(figsize=(10, 5))
sns.boxplot(x=cv_results["param_max_depth"], y=cv_results["mean_test_score"])
plt.xlabel("Max Depth")
plt.ylabel("Mean Test Score (R²)")
plt.title("Effect of max_depth on Model Performance")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=cv_results["param_min_samples_split"], y=cv_results["mean_test_score"])
plt.xlabel("Min Samples Split")
plt.ylabel("Mean Test Score (R²)")
plt.title("Effect of min_samples_split on Model Performance")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=cv_results["param_min_samples_leaf"], y=cv_results["mean_test_score"])
plt.xlabel("Min Samples Leaf")
plt.ylabel("Mean Test Score (R²)")
plt.title("Effect of min_samples_leaf on Model Performance")
plt.show()

plt.figure(figsize=(10, 5))

# Ensure max_features is a string for Seaborn
cv_results["param_max_features"] = cv_results["param_max_features"].astype(str)

sns.boxplot(x=cv_results["param_max_features"], y=cv_results["mean_test_score"])
plt.xlabel("Max Features")
plt.ylabel("Mean Test Score (R²)")
plt.title("Effect of max_features on Model Performance")
plt.show()

plt.figure(figsize=(10, 5))

# Convert bootstrap to string for Seaborn
cv_results["param_bootstrap"] = cv_results["param_bootstrap"].astype(str)

sns.boxplot(x=cv_results["param_bootstrap"], y=cv_results["mean_test_score"])
plt.xlabel("Bootstrap")
plt.ylabel("Mean Test Score (R²)")
plt.title("Effect of Bootstrap on Model Performance")
plt.show()

# Convert search results to a DataFrame
cv_results = pd.DataFrame(rf_random.cv_results_)

# Select only relevant columns
cv_results = cv_results[["mean_test_score", "param_n_estimators", "param_max_depth", 
                         "param_min_samples_split", "param_min_samples_leaf", "param_max_features",
                         "param_bootstrap"]]

# Sort by best test score
cv_results = cv_results.sort_values(by="mean_test_score", ascending=False)

cv_results

import matplotlib.pyplot as plt
import pandas as pd

# Get feature importances from the best model
feature_importances = best_model.feature_importances_

# Get feature names after encoding
feature_names = X.columns

# Convert feature importances into a DataFrame
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Define categorical features before encoding
categorical_features = ["Standort FZ", "Wochentag", "Monat"]  # Original categorical columns

# Dictionary to store aggregated feature importances
aggregated_importances = {}

# Aggregate feature importances for each categorical variable
for cat in categorical_features:
    # Find all one-hot encoded feature names that start with the categorical feature name
    matching_features = importance_df[importance_df['Feature'].str.startswith(cat + "_")]

    # Sum their importance scores
    aggregated_importances[cat] = matching_features["Importance"].sum()

# Keep numerical and boolean features unchanged
numerical_features = [f for f in feature_names if not any(f.startswith(cat + "_") for cat in categorical_features)]
for num in numerical_features:
    aggregated_importances[num] = importance_df.loc[importance_df["Feature"] == num, "Importance"].sum()

# Convert back to DataFrame and sort
final_importance_df = pd.DataFrame.from_dict(aggregated_importances, orient='index', columns=['Importance'])
final_importance_df = final_importance_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(final_importance_df.index, final_importance_df['Importance'], edgecolor='black')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.title('Aggregated Feature Importances in RandomForestRegressor')
plt.gca().invert_yaxis()  # Highest importance on top
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Number of top models to consider
top_n = 10

# Get the indices of the top N models based on cross-validation score
best_indices = np.argsort(rf_random.cv_results_['mean_test_score'])[-top_n:]  

# Extract feature names
feature_names = X.columns

# Define categorical features before encoding
categorical_features = ["Standort FZ", "Wochentag", "Monat"] 

# Store feature importances for each of the top models
importances_list = []

for idx in best_indices:
    model_params = rf_random.cv_results_['params'][idx]  # Get hyperparameters
    rf_model = RandomForestRegressor(**model_params, random_state=42, n_jobs=-1)  # Recreate the model
    rf_model.fit(X, y)  # Train on full data
    importances_list.append(rf_model.feature_importances_)  # Store feature importances

# Convert to DataFrame for easier manipulation
importances_df = pd.DataFrame(importances_list, columns=feature_names)

# Compute mean and standard deviation
mean_importances = importances_df.mean(axis=0)
std_importances = importances_df.std(axis=0)

# Aggregate feature importances for categorical variables
aggregated_mean_importances = {}
aggregated_std_importances = {}

for cat in categorical_features:
    matching_features = [f for f in feature_names if f.startswith(cat + "_")]
    aggregated_mean_importances[cat] = mean_importances[matching_features].sum()
    aggregated_std_importances[cat] = np.sqrt(np.sum(std_importances[matching_features]**2))  # Error propagation

# Keep numerical/boolean features unchanged
numerical_features = [f for f in feature_names if not any(f.startswith(cat + "_") for cat in categorical_features)]
for num in numerical_features:
    aggregated_mean_importances[num] = mean_importances[num]
    aggregated_std_importances[num] = std_importances[num]

# Convert to DataFrame and sort
final_importance_df = pd.DataFrame({
    'Mean Importance': aggregated_mean_importances,
    'Std Importance': aggregated_std_importances
}).sort_values(by='Mean Importance', ascending=False)

# Plot mean feature importances with error bars
plt.figure(figsize=(10, 6))
plt.barh(final_importance_df.index, final_importance_df['Mean Importance'], 
         xerr=final_importance_df['Std Importance'], capsize=5, edgecolor='black')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.title(f'Mean Feature Importance Over Top {top_n} Models (with Std Dev)')
plt.gca().invert_yaxis()  # Highest importance on top
plt.show()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define parameter grid
param_dist = {
    'n_estimators': randint(50, 500),  # Number of trees
    'max_depth': [20, 30, None],  # Tree depth
    'min_samples_split': [15, 20, 10],  # Minimum samples to split
    'min_samples_leaf': [2, 4, 6, 8],  # Minimum samples per leaf
    'max_features': ['sqrt'],  # Features to consider per split
    'bootstrap': [True, False]  # Bootstrapping
}

# Define the model
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# Define scoring metrics
scoring = {
    'R2': 'r2',
    'RMSE': 'neg_root_mean_squared_error'
}

# Perform Randomized Search
rf_random = RandomizedSearchCV(
    estimator=rf,  # Directly optimize RandomForestRegressor, NOT another RandomizedSearchCV
    param_distributions=param_dist,
    n_iter=10,  # Number of different combinations to try
    cv=5,  # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring=scoring,
    refit='RMSE'  # Refit using RMSE
)

# Fit the model
rf_random.fit(X, y)

# Convert search results to a DataFrame
cv_results = pd.DataFrame(rf_random.cv_results_)
cv_results

importances = rf_random.best_estimator_.feature_importances_
print(sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True))

