#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import joblib
os.nice(20)

data = pd.read_parquet("../data/interim/selected-data_with_proper_time.parquet")
print("\n=== Merged Dataset ===")
print("Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nMerged DataFrame info:")
print(data.info())

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_parquet("../data/interim/selected-data_with_proper_time.parquet")

data['MELDUNGSEINGANG'] = pd.to_datetime(data['MELDUNGSEINGANG'])

# 2. Weekly aggregation
weekly = (
    data
    .groupby(data['MELDUNGSEINGANG'].dt.to_period('W').apply(lambda r: r.start_time))
    .size()
    .reset_index(name='Count')
    .rename(columns={'MELDUNGSEINGANG': 'WeekStart'})
)
weekly['Ordinal'] = weekly['WeekStart'].map(lambda d: d.toordinal())

# 3. Fit linear trend and get covariance
#    coeffs = [slope, intercept]; cov is 2×2 covariance matrix
coeffs, cov = np.polyfit(weekly['Ordinal'], weekly['Count'], 1, cov=True)
m, b = coeffs
m_err, b_err = np.sqrt(np.diag(cov))

# 4. Print with uncertainties
print(f"Slope (m): {365*m:.4f} ± {365*m_err:.4f} Einsätze per week-ordinal")
print(f"Intercept (b): {b:.2f} ± {b_err:.2f} Einsätze")

# 5. Plot
plt.figure(figsize=(6, 4))
plt.plot(weekly['WeekStart'], weekly['Count'], marker='o', linestyle='',
         label='Wöchentliche Einsätze')
plt.plot(weekly['WeekStart'], m * weekly['Ordinal'] + b,
         label=f'Trend')
plt.xlabel('Jahr')
plt.ylabel('Anzahl Einsätze')
plt.title('Einsätze pro Woche')
plt.legend()
plot_output_path = "../reports/figures/1.00-weekly-trend.png"
plt.savefig(plot_output_path, dpi=300)
plt.tight_layout()
plt.show()

m * weekly['Ordinal'] + b

# In[ ]:

