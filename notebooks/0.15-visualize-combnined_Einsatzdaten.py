#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from src.load import load_combined_einsatzdaten
from src.create_interim_data import load_Einsatzdaten_2023, load_Einsatzdaten_2018_22

data1 = load_Einsatzdaten_2018_22()
data2 =  load_Einsatzdaten_2023()
data= load_combined_einsatzdaten(force=True)

data2.info()

data1.info()

data.info()

data.describe()

import pandas as pd
import matplotlib.pyplot as plt

# Ensure 'MELDUNGSEINGANG' is a datetime type
data['MELDUNGSEINGANG'] = pd.to_datetime(data['MELDUNGSEINGANG'], errors='coerce')

# Drop missing values and filter for dates from 2018 onwards
meldungen_data = data['MELDUNGSEINGANG'].dropna()
meldungen_data = meldungen_data[meldungen_data >= "2018-01-01"]

# Group by continuous weeks
weekly_counts = (
    meldungen_data.groupby(meldungen_data.dt.to_period('W'))
    .size()
    .sort_index()
)

# Convert PeriodIndex to datetime for plotting
weekly_counts.index = weekly_counts.index.to_timestamp()

# Plot the weekly aggregated data
plt.figure(figsize=(12, 6))
plt.plot(
    weekly_counts.index, 
    weekly_counts.values, 
    color='skyblue', 
    marker='o', 
    linestyle='-', 
    linewidth=1
)
plt.title("Einsätze pro Woche")
plt.xlabel("Jahr")
plt.ylabel("Anzahl")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save the figure
output_path = "../reports/figures/einsaetze_pro_woche"
plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_path}.eps", format='eps', bbox_inches='tight')

# Show the plot
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Ensure 'MELDUNGSEINGANG' is a datetime type
data['MELDUNGSEINGANG'] = pd.to_datetime(data['MELDUNGSEINGANG'], errors='coerce')

# Drop missing values and filter for dates from 2018 onwards
meldungen_data = data['MELDUNGSEINGANG'].dropna()
meldungen_data = meldungen_data[meldungen_data >= "2018-01-01"]

# Group by continuous days
daily_counts = (
    meldungen_data.groupby(meldungen_data.dt.to_period('D'))
    .size()
    .sort_index()
)

# Convert PeriodIndex to datetime for plotting
daily_counts.index = daily_counts.index.to_timestamp()

# Plot the daily aggregated data
plt.figure(figsize=(12, 6))
plt.scatter(
    daily_counts.index, 
    daily_counts.values, 
    color='skyblue', 
    marker='o'
)
plt.title("Einsätze pro Tag")
plt.xlabel("Zeit")
plt.ylabel("Anzahl")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save the figure
output_path = "../reports/figures/einsaetze_pro_tag"
plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_path}.eps", format='eps', bbox_inches='tight')

# Show the plot
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Drop missing values in 'ALARMZEIT' if any, and extract the hour
alarmzeit_hours = data['ALARMZEIT'].dropna().dt.hour

# Plot the histogram for the time of day distribution
plt.figure(figsize=(10, 6))
plt.hist(
    alarmzeit_hours, 
    bins=24, 
    range=(0, 24), 
    color='salmon', 
    edgecolor='black'
)
plt.title("Einsätze nach Uhrzeit")
plt.xlabel("Uhrzeit")
plt.ylabel("Anzahl")
plt.xticks(range(0, 24))
plt.tight_layout()

# Save the figure
output_path = "../reports/figures/einsaetze_nach_uhrzeit"
plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_path}.eps", format='eps', bbox_inches='tight')

# Show the plot
plt.show()

# Drop missing values in 'MELDUNGSEINGANG' and extract the hour
meldung_hours = data['MELDUNGSEINGANG'].dropna().dt.hour

# Plot the histogram for the time of day distribution
plt.figure(figsize=(10, 6))
plt.hist(meldung_hours, bins=24, range=(0, 24), color='skyblue', edgecolor='black')
plt.title("Distribution of 'MELDUNGSEINGANG' by Time of Day")
plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Frequency")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Drop missing values and extract the day of the week from 'ALARMZEIT'
alarmzeit_days = data['ALARMZEIT'].dropna().dt.day_name()

# Manuelle Übersetzung der englischen Wochentage ins Deutsche
tage_deutsch = {
    "Monday": "Montag", "Tuesday": "Dienstag", "Wednesday": "Mittwoch",
    "Thursday": "Donnerstag", "Friday": "Freitag", "Saturday": "Samstag", "Sunday": "Sonntag"
}
alarmzeit_days = alarmzeit_days.map(tage_deutsch)

# Count occurrences for each day of the week
day_counts = alarmzeit_days.value_counts().reindex(
    ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
)

# Plot the distribution
plt.figure(figsize=(10, 6))
day_counts.plot(kind='bar', color='skyblue', edgecolor='black', width=0.8)  # Balken dichter zusammen
plt.title("Einsätze nach Wochentag")
plt.xlabel("Wochentag")
plt.ylabel("Anzahl")
plt.xticks(rotation=45)
plt.tight_layout()

# Save the figure
output_path = "../reports/figures/einsaetze_nach_wochentag"
plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_path}.eps", format='eps', bbox_inches='tight')

# Show the plot
plt.show()

# Drop missing values and extract the day of the week from 'MELDUNGSEINGANG'
meldungseingang_days = data['MELDUNGSEINGANG'].dropna().dt.day_name()

# Count occurrences for each day of the week
meldungseingang_day_counts = meldungseingang_days.value_counts().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

# Plot the distribution
plt.figure(figsize=(10, 6))
meldungseingang_day_counts.plot(kind='bar', color='salmon', edgecolor='black')
plt.title("Distribution of 'MELDUNGSEINGANG' by Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Count values for "J" (Yes) and "N" (No), ignoring NaN if it is 0
sondersignal_counts = data['SONDERSIGNAL'].value_counts(dropna=False)
sizes = [sondersignal_counts.get(True, 0), sondersignal_counts.get(False, 0)]
labels = ['Ja', 'Nein']

# Falls die NA-Kategorie nicht 0 ist, fügen wir sie hinzu
na_count = sondersignal_counts.isna().sum()
if na_count > 0:
    sizes.append(na_count)
    labels.append('NA')

# Plot a pie chart
plt.figure(figsize=(8, 6))
plt.pie(
    sizes, 
    labels=labels, 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=['lightgreen', 'lightcoral', 'lightgray']
)
plt.title("Anteil der Einsätze mit Sondersignal")

# Save the figure
output_path = "../reports/figures/einsaetze_mit_sondersignal"
plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_path}.eps", format='eps', bbox_inches='tight')

# Show the plot
plt.show()

import folium

# Calculate the center of the map based on the mean of latitudes and longitudes
center_lat = data['EINSATZORT_lat'].mean()
center_lon = data['EINSATZORT_lon'].mean()

# Initialize the Folium map centered on the average location
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Add points to the map
for lat, lon in zip(data['EINSATZORT_lat'], data['EINSATZORT_lon']):
    if not pd.isna(lat) and not pd.isna(lon):  # Ensure there are no NaN values
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,  # Small radius for points
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            opacity=0.6
        ).add_to(m)
m.save("../reports/figures/Einsatzort_map.html")

# Initialize the Folium map centered on the average location
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Add points to the map
for lat, lon in zip(data['ZIELORT_lat'], data['ZIELORT_lon']):
    if not pd.isna(lat) and not pd.isna(lon):  # Ensure there are no NaN values
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,  # Small radius for points
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            opacity=0.6
        ).add_to(m)
m.save("../reports/figures/Zielort_map.html")

import numpy as np
import matplotlib.pyplot as plt

# Calculate 'eintreffzeit' in minutes
data['eintreffzeit'] = (data['Status 4'] - data['MELDUNGSEINGANG']).dt.total_seconds() / 60  # Convert to minutes

# Filter for eintreffzeit values in the range (0, 600] minutes
data['eintreffzeit'] = data['eintreffzeit'].apply(lambda x: x if 0 < x <= 600 else np.nan)

# Check counts for each 'SONDERSIGNAL' value
print("Counts of 'eintreffzeit' in the range 0-600 minutes:")
print(data.groupby('SONDERSIGNAL')['eintreffzeit'].count())

# Plot distribution of 'eintreffzeit' values within the 0-600 minute range
plt.figure(figsize=(10, 6))
plt.hist(data['eintreffzeit'].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of 'eintreffzeit' (0-600 minutes)")
plt.xlabel("eintreffzeit (minutes)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Plot distribution conditioned on 'SONDERSIGNAL' for 'eintreffzeit' in the 0-600 minute range
plt.figure(figsize=(10, 6))
for sondersignal_value, color in zip([True, False], ['lightgreen', 'lightcoral']):
    subset = data[(data['SONDERSIGNAL'] == sondersignal_value) & (data['eintreffzeit'] <= 60)]['eintreffzeit'].dropna()
    if not subset.empty:
        plt.hist(subset, bins=30, alpha=0.6, color=color, edgecolor='black', label=f"SONDERSIGNAL = {sondersignal_value}")

plt.title("Distribution of 'eintreffzeit' (0-600 minutes) Conditioned on SONDERSIGNAL")
plt.xlabel("eintreffzeit (minutes)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

data["eintreffzeit"].max()

((data['Status 4'] - data['MELDUNGSEINGANG']).dt.total_seconds() / 60).max()

((data['Status 4'] - data['MELDUNGSEINGANG']).dt.total_seconds() / 60).min()

((data['Status 4'] - data['MELDUNGSEINGANG']).dt.total_seconds() / 60)

import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Filter data for the specified conditions, including non-NaN lat and lon
filtered_data = data[
    (data['SONDERSIGNAL'] == True) & 
    (data['eintreffzeit'] > 0) & 
    (data['eintreffzeit'] <= 120) & 
    (~data['EINSATZORT_lat'].isna()) & 
    (~data['EINSATZORT_lon'].isna())
]

# Calculate the center of the map based on the mean of latitudes and longitudes
center_lat = filtered_data['EINSATZORT_lat'].mean()
center_lon = filtered_data['EINSATZORT_lon'].mean()

# Initialize the Folium map centered on the average location
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Transform 'eintreffzeit' using log(eintreffzeit + 1)
filtered_data['log_eintreffzeit'] = np.log(filtered_data['eintreffzeit'] + 1)

# Set up a colormap for the transformed eintreffzeit (log scale)
colormap = plt.cm.get_cmap('RdYlGn_r')  # Inverted colormap (green for short, red for long)
normalize = mcolors.Normalize(vmin=np.log( 2), vmax=np.log(120 + 1))

# Add points to the map with color-coding based on log-transformed eintreffzeit
for lat, lon, eintreff_log in zip(filtered_data['EINSATZORT_lat'], filtered_data['EINSATZORT_lon'], filtered_data['log_eintreffzeit']):
    # Map log-transformed eintreffzeit to a color
    color = mcolors.to_hex(colormap(normalize(eintreff_log)))
    
    # Add the CircleMarker with the color-coded eintreffzeit
    folium.CircleMarker(
        location=[lat, lon],
        radius=3,  # Small radius for points
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        opacity=0.7
    ).add_to(m)

# Save the map to an HTML file
output_path = "../plots/figures/Einsatzort_map_log_colorcoded.html"
#m.save(output_path)
#print(f"Map saved to {output_path}")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
import numpy as np
# Define latitude and longitude boundaries
min_lon, min_lat = 7.37586644151297, 51.264836728429444
max_lon, max_lat = 7.598894722085717, 51.41859378414624

# Filter data for specified conditions, including lat/lon boundaries
filtered_data = data[
    (data['SONDERSIGNAL'] == True) & 
    (data['eintreffzeit'] > 0) & 
    (data['eintreffzeit'] <= 120) & 
    (~data['EINSATZORT_lat'].isna()) & 
    (~data['EINSATZORT_lon'].isna()) &
    (data['EINSATZORT_lat'] >= min_lat) & (data['EINSATZORT_lat'] <= max_lat) &
    (data['EINSATZORT_lon'] >= min_lon) & (data['EINSATZORT_lon'] <= max_lon)
]
filtered_data['log_eintreffzeit'] = np.log(filtered_data['eintreffzeit'] + 1)

# Convert lat/lon to GeoDataFrame
geometry = [Point(lon, lat) for lon, lat in zip(filtered_data['EINSATZORT_lon'], filtered_data['EINSATZORT_lat'])]
gdf = gpd.GeoDataFrame(filtered_data, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)

# Set up a colormap and normalize for log_eintreffzeit
colormap = plt.cm.get_cmap('nipy_spectral')
normalize = mcolors.Normalize(vmin=0, vmax=np.log(120 + 1))

# Calculate the bounding box with buffer
buffer_distance = 500  # Adjust as needed
minx, miny, maxx, maxy = gdf.total_bounds
buffer_extent = [minx - buffer_distance, miny - buffer_distance, maxx + buffer_distance, maxy + buffer_distance]

# Plot using Matplotlib
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color=[mcolors.to_hex(colormap(normalize(val))) for val in gdf['log_eintreffzeit']], markersize=10, alpha=0.7)

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=14)

# Set the x and y limits to the buffered bounding box
ax.set_xlim(buffer_extent[0], buffer_extent[2])
ax.set_ylim(buffer_extent[1], buffer_extent[3])

# Set up the plot appearance
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')

# Save and display the plot
plt.tight_layout(pad=0)
output_path = "../reports/figures/einsatzort_map_static.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig)  # This prevents display in Jupyter
print(f"Map saved to {output_path}")

# In[ ]:

