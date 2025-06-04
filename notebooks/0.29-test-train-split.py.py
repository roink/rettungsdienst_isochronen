#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd

data = pd.read_parquet("../data/interim/selected-data.parquet")
print("\n=== Merged Dataset ===")
print("Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nMerged DataFrame info:")
print(data.info())

from sklearn.model_selection import train_test_split

# Datensatz in Trainings- (80%) und Test-Daten (20%) aufteilen
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

output_path = os.path.join("../data/interim", "train_dataset.parquet")
train_df.to_parquet(output_path, index=False)
output_path = os.path.join("../data/interim", "test_dataset.parquet")
test_df.to_parquet(output_path, index=False)

