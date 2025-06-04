#!/usr/bin/env python
# coding: utf-8

import pandas as pd

data = pd.read_parquet("../data/interim/selected-data.parquet")

data.rename(columns={'Eintreffzeit': 'Hilfsfrist'}, inplace=True)
data.info()

import numpy as np
import matplotlib.pyplot as plt

# Integer x values from 0 to 20
x_values = np.arange(0, 21)

# Compute cumulative percentage
percentages = [np.mean(data['Hilfsfrist'] <= x) * 100 for x in x_values]

# Success rates at 8 and 12 minutes
success_8 = np.mean(data['Hilfsfrist'] <= 8) * 100
success_12 = np.mean(data['Hilfsfrist'] <= 12) * 100

# Plot
plt.figure(figsize=(6, 4))
plt.plot(x_values, percentages, marker='o', linestyle='-', color='blue')

# Horizontal line at 90%
plt.axhline(y=90, color='grey', linestyle='--', label='90% threshold')

plt.xlim(0, 20)
plt.ylim(0, 100)

# Get current x-axis range
x_min, x_max = plt.xlim()

# Normalize the x-values (8 and 12) to the range [0, 1]
plt.axhline(y=success_8, xmin=0, xmax=8 / (x_max - x_min), color='grey', linestyle='--')
plt.axhline(y=success_12, xmin=0, xmax=12 / (x_max - x_min), color='grey', linestyle='--')

# Vertical lines at 8 and 12 minutes
plt.axvline(x=8,ymin=0,ymax=success_8/100, color='grey', linestyle='--')
plt.axvline(x=12,ymax=success_12/100, color='grey', linestyle='--')

# Annotate success rates
plt.text(7, success_8 , f'{success_8:.1f}%', ha='center', va='bottom', color='black')
plt.text(11, success_12 , f'{success_12:.1f}%', ha='center', va='bottom', color='black')
plt.text(4, 91 , f'90% Zielvorgabe', ha='center', va='bottom', color='black')

# Formatting
plt.xticks(np.arange(0, 21, 2))
plt.xlim(0, 20)
plt.ylim(0, 100)
plt.xlabel('Hilfsfrist [min]')
plt.ylabel('Kumulativer Anteil [%]')
plt.title('Kumulative Verteilung der Hilfsfrist')
plt.grid(True)
plt.tight_layout()
plt.savefig("../reports/figures/1.12-Kumulative-Hilfsfrist.png",dpi=300)
plt.show()

# Zeitachse (von 0 bis 20 Minuten)
max_time = 20
x_values = np.arange(0, max_time + 1)

# Kumulativer Anteil für jede Komponente berechnen
disp_pct = [np.mean(data['Dispositionszeit'] <= x) * 100 for x in x_values]
ausr_pct = [np.mean(data['Ausrückzeit']      <= x) * 100 for x in x_values]
fahr_pct = [np.mean(data['Fahrzeit']          <= x) * 100 for x in x_values]
Hilfsfris_pct = [np.mean(data['Hilfsfrist']          <= x) * 100 for x in x_values]

# Ergebnisse in DataFrame packen
df_cumulative = pd.DataFrame({
    'Zeit [min]': x_values,
    'Gesprächs- und Dispositionszeit [%]': disp_pct,
    'Ausrückzeit [%]': ausr_pct,
    'Fahrzeit [%]': fahr_pct,
    'Hilfsfrist [%]': Hilfsfris_pct
  
})

# Plot erstellen
plt.figure(figsize=(6,4))
plt.plot(x_values, disp_pct, marker='o', linestyle='-',  label='Gesprächs- und Dispositionszeit')
plt.plot(x_values, ausr_pct, marker='s', linestyle='--', label='Ausrückzeit')
plt.plot(x_values, fahr_pct, marker='^', linestyle='-.', label='Fahrzeit')

# 90%-Schwelle
#plt.axhline(90, color='grey', linestyle='--', label='90% Schwelle')

# Achsengrenzen und Beschriftungen
plt.xlim(0, max_time)
plt.ylim(0, 100)
plt.xlabel('Zeit [min]')
plt.ylabel('Kumulativer Anteil [%]')
plt.title('Kumulative Verteilung der Zeitkomponenten')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

# Anzeige und optionales Speichern\# 
plt.savefig("../reports/figures/1.12-kumulative_zeitkomponenten.png", dpi=300)
plt.show()

df_cumulative.round(1)

import matplotlib.pyplot as plt
import pandas as pd

# Filter the data for Eintreffzeit values between 0 and 20
filtered = data[(data['Hilfsfrist'] >= 0) & (data['Hilfsfrist'] <= 20)]

# Group by Eintreffzeit and compute the mean of each component
grouped = filtered.groupby('Hilfsfrist').agg({
    'Dispositionszeit': 'mean',
    'Ausrückzeit': 'mean',
    'Fahrzeit': 'mean'
})

# Reindex to ensure that all x values from 0 to 20 are present (fill missing with 0)
grouped = grouped.reindex(range(0, 21), fill_value=0)

# Prepare x axis (Eintreffzeit values)
x = grouped.index

plt.figure(figsize=(6, 4))

# Create the stacked bar chart
plt.bar(x, grouped['Dispositionszeit'], label='Gesprächs- und Dispositionszeit')
plt.bar(x, grouped['Ausrückzeit'], bottom=grouped['Dispositionszeit'], label='Ausrückzeit')
# Compute the bottom for Fahrzeit by adding the previous two averages
bottoms = grouped['Dispositionszeit'] + grouped['Ausrückzeit']
plt.bar(x, grouped['Fahrzeit'], bottom=bottoms, label='Fahrzeit')

plt.xlabel('Hilfsfrist [min]')
plt.ylabel('Durchschnittliche Zeit [min]')
plt.title('Absolute mittlere Zeitanteile')
plt.legend()
plt.xlim(0, 20)
plt.xticks(np.arange(0, 21, 2))
print(grouped)
plt.tight_layout()
plt.savefig("../reports/figures/1.12-Absolute-mittlere-Zeitanteile.png",dpi=300)
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Filter the data for Eintreffzeit values between 0 and 20
filtered = data[(data['Hilfsfrist'] >= 0) & (data['Hilfsfrist'] <= 20)]

# Group by Eintreffzeit and compute the mean of each component
grouped = filtered.groupby('Hilfsfrist').agg({
    'Dispositionszeit': 'mean',
    'Ausrückzeit': 'mean',
    'Fahrzeit': 'mean'
})

# Reindex to ensure that all x values from 0 to 20 are present (fill missing with 0)
grouped = grouped.reindex(range(0, 21), fill_value=0)

# Calculate total for each row
total = grouped.sum(axis=1)

# Avoid division by zero
total[total == 0] = np.nan

# Convert to relative percentages
percentages = grouped.divide(total, axis=0) * 100

# Prepare x axis (Eintreffzeit values)
x = percentages.index
plt.figure(figsize=(6, 4))

# Create the stacked bar chart with percentages
plt.bar(x, percentages['Dispositionszeit'], label='Gesprächs- und Dispositionszeit')
plt.bar(x, percentages['Ausrückzeit'], bottom=percentages['Dispositionszeit'], label='Ausrückzeit')
bottoms = percentages['Dispositionszeit'] + percentages['Ausrückzeit']
plt.bar(x, percentages['Fahrzeit'], bottom=bottoms, label='Fahrzeit')

plt.xlabel('Hilfsfrist [min]')
plt.ylabel('Anteil [%]')
plt.title('Relative mittlere Zeitanteile')
plt.legend(loc='upper right')
plt.xlim(0, 20)
plt.ylim(0, 100)
plt.xticks(np.arange(0, 21, 2))
print(percentages)
plt.tight_layout()
plt.savefig("../reports/figures/1.12-Relative-mittlere-Zeitanteile.png",dpi=300)
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

# Select the relevant columns
cols = ['Dispositionszeit', 'Ausrückzeit', 'Fahrzeit', 'Hilfsfrist']
corr_matrix = data[cols].corr()

# Create a mask to show only one triangle 
#mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

wrapped_labels = [
    'Dispositions-\nzeit',
    'Ausrück-\nzeit',
    'Fahrzeit',
    'Hilfsfrist'
]

plt.figure(figsize=(7, 6))
ax = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    square=True,
    linewidths=.5,
    cbar_kws={"shrink": .75}
)

# Make x-labels horizontal and adjust font size
ax.set_xticklabels(
    wrapped_labels,
    rotation=0,
    ha="center",
    fontsize=12
)
# Align y-labels to the right for consistency
ax.set_yticklabels(
    wrapped_labels,
    rotation=0,
    va="center",
    ha="right",
    fontsize=12
)

plt.title('Korrelations-Matrix', pad=12, fontsize=14)
plt.tight_layout()
plt.savefig("../reports/figures/1.12-Korrelationsmatrix.png", dpi=300)
plt.show()

vars = ['Dispositionszeit', 'Ausrückzeit', 'Fahrzeit']
X = data[vars]
Z = data['Hilfsfrist']

# Variance and covariance matrix
cov = X.cov()
var_Z = Z.var()

# Total variance from sum
var_components = {
    var: X[var].var() for var in vars
}
cov_components = {
    f"{i}×{j}": 2 * cov.loc[i, j] for i in vars for j in vars if i < j
}

# Combine and normalize
total_explained_variance = sum(var_components.values()) + sum(cov_components.values())
explained_ratio = {
    k: v / var_Z for k, v in {**var_components, **cov_components}.items()
}

# Display explained contribution as percent of Eintreffzeit variance
for k, v in explained_ratio.items():
    print(f"{k}: {v*100:.2f}% of Hilfsfrist variance")

for col in ['Dispositionszeit', 'Ausrückzeit', 'Fahrzeit']:
    r = data['Hilfsfrist'].corr(data[col])
    print(f"{col}: R² = {r**2:.2f}")

import statsmodels.api as sm

X = data[['Dispositionszeit', 'Ausrückzeit', 'Fahrzeit']]
y = data['Hilfsfrist']

# Add intercept and fit OLS
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

