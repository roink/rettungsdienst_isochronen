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

# 1. Extrahiere die Stunde aus dem Zeitstempel
data['Stunde'] = data['MELDUNGSEINGANG'].dt.hour

# 2. Zähle die Einsätze je Stunde
einsatz_pro_stunde = (
    data
    .groupby('Stunde')
    .size()
    .reset_index(name='Anzahl_Einsätze')
    .sort_values('Stunde')
)

print(einsatz_pro_stunde)

# 3. Balkendiagramm zur Visualisierung
plt.figure(figsize=(6, 4))
plt.bar(einsatz_pro_stunde['Stunde'], einsatz_pro_stunde['Anzahl_Einsätze'])
plt.xlabel('Stunde des Tages')
plt.ylabel('Anzahl der Einsätze')
plt.title('Einsätze pro Stunde')
plt.xticks(range(24))
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Daten einlesen
data = pd.read_parquet("../data/interim/selected-data_with_proper_time.parquet")

# Datum und Stunde extrahieren
data['Datum'] = pd.to_datetime(data['MELDUNGSEINGANG'].dt.date)
data['Stunde'] = data['MELDUNGSEINGANG'].dt.hour

# 1. Alle Kalendertage im Zeitraum erzeugen
start_date = data['Datum'].min()
end_date = data['Datum'].max()
all_dates = pd.date_range(start_date, end_date, freq='D')

# 2. Tägliche Einsatzzahl pro Stunde berechnen
daily_hourly = (
    data
    .groupby(['Datum', 'Stunde'])
    .size()
    .reset_index(name='Anzahl_Einsätze')
    .pivot(index='Datum', columns='Stunde', values='Anzahl_Einsätze')
    .reindex(all_dates, fill_value=0)  # fehlende Tage auffüllen
)

daily_hour= daily_hourly.fillna(0)

# 3. Statistik pro Stunde berechnen
summary_hour = daily_hourly.agg(['sum', 'mean', 'std']).T
summary_hour['sem'] = summary_hour['std'] / np.sqrt(summary_hour['sum'])
summary_hour = summary_hour.reset_index().rename(columns={'index': 'Stunde'})

summary_hour

# 5. Balkendiagramm mit Fehlerbalken (SEM)
plt.figure(figsize=(8, 4))
plt.bar(summary_hour['Stunde'], summary_hour['mean'], yerr=summary_hour['sem'], capsize=5)
plt.xlabel('Stunde des Tages')
plt.ylabel('Durchschnittliche Einsätze pro Kalendertag')
plt.title('Normalisierte Einsätze pro Stunde mit Fehlerbalken (SEM)')
plt.xticks(range(24))
plt.tight_layout()
plt.show()

# 6. Boxplot der Verteilung pro Stunde
plt.figure(figsize=(10, 5))
data_for_box = [daily_hourly[col] for col in sorted(daily_hourly.columns)]
plt.boxplot(data_for_box, labels=sorted(daily_hourly.columns))
plt.xticks(rotation=45)
plt.xlabel('Stunde des Tages')
plt.ylabel('Tägliche Einsätze')
plt.title('Verteilung der Einsätze pro Stunde')
plt.tight_layout()
plt.show()

# 7. Beispiel-Statistik für Stunde 8
print("Verteilungs-Statistik für Einsätze um 08:00 Uhr:")
print(daily_hourly[8].describe())

summary

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Daten einlesen
data = pd.read_parquet("../data/interim/selected-data_with_proper_time.parquet")

# Datum und Wochentag extrahieren
data['Datum'] = data['MELDUNGSEINGANG'].dt.date
data['Wochentag'] = data['MELDUNGSEINGANG'].dt.day_name()

# Tägliche Einsatzzahl ermitteln
daily_counts = (
    data
    .groupby(['Datum', 'Wochentag'])
    .size()
    .reset_index(name='Anzahl_Einsätze')
)

# Statistik pro Wochentag berechnen
weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
summary = (
    daily_counts
    .groupby('Wochentag')['Anzahl_Einsätze']
    .agg(count='count', mean='mean', std='std')
)
summary['sem'] = summary['std'] / np.sqrt(summary['count'])
summary = summary.reindex(weekday_order)

# Tabelle mit Statistik anzeigen

# Bar-Chart mit Fehlerbalken (SEM)
plt.figure()
x = np.arange(len(weekday_order))
plt.bar(x, summary['mean'], yerr=summary['sem'], capsize=5)
plt.xticks(x, weekday_order, rotation=45)
plt.xlabel('Wochentag')
plt.ylabel('Durchschnittliche Einsätze pro Tag')
plt.title('Normalisierte Einsätze mit Fehlerbalken (SEM)')
plt.tight_layout()
plt.show()

# Boxplot der Verteilung pro Wochentag
plt.figure()
data_for_box = [daily_counts.loc[daily_counts['Wochentag'] == wd, 'Anzahl_Einsätze'] for wd in weekday_order]
plt.boxplot(data_for_box, labels=weekday_order)
plt.xticks(rotation=45)
plt.ylim(0,85)
plt.xlabel('Wochentag')
plt.ylabel('Tägliche Einsätze')
plt.title('Verteilung der Einsätze pro Wochentag')
plt.tight_layout()
plt.show()

summary

# In[ ]:

