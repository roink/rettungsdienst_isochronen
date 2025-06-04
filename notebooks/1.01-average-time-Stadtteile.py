#!/usr/bin/env python
# coding: utf-8

import os
import sys
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import contextily as ctx

sys.path.insert(0, os.path.abspath("../src"))
from plotting import plot_shapefile_with_labels

stadtbezirke_path = "../data/raw/Stadtbezirke_Hagen.shp"
statistische_bezirke_path = "../data/raw/Statistische_Bezirke_Hagen.shp"
wohnbezirke_path = "../data/raw/Wohnbezirke_Hagen.shp"
final_dataset_path = "../data/interim/selected-data.parquet"

points_df = pd.read_parquet(final_dataset_path)
points_gdf = gpd.GeoDataFrame(
    points_df,
    geometry=gpd.points_from_xy(points_df.EINSATZORT_lon, points_df.EINSATZORT_lat),
    crs="EPSG:4326"
)

# Function to calculate average Eintreffzeit per district
def calculate_average_eintreffzeit(district_path, points_gdf, name_column):
    districts_gdf = gpd.read_file(district_path)
    if districts_gdf.crs != "EPSG:4326":
        districts_gdf = districts_gdf.to_crs(epsg=4326)
    joined_gdf = gpd.sjoin(points_gdf, districts_gdf, how="inner", predicate="within")
    avg_eintreffzeit = joined_gdf.groupby(name_column)["Eintreffzeit"].mean().reset_index()
    districts_gdf = districts_gdf.merge(avg_eintreffzeit, on=name_column, how="left")
    return districts_gdf

# Calculate averages for each district level
stadtbezirke_gdf = calculate_average_eintreffzeit(stadtbezirke_path, points_gdf, "BEZEICHNUN")
statistische_bezirke_gdf = calculate_average_eintreffzeit(statistische_bezirke_path, points_gdf, "BEZEICHNUN")
wohnbezirke_gdf = calculate_average_eintreffzeit(wohnbezirke_path, points_gdf, "NAME")

stadtbezirke_gdf[['BEZEICHNUN', 'Eintreffzeit']].sort_values('Eintreffzeit').rename(
    columns={'BEZEICHNUN': 'Stadtbezirk', 'Eintreffzeit': 'Durchschnittliche Eintreffzeit'}
)

statistische_bezirke_gdf[['BEZEICHNUN', 'Eintreffzeit']].sort_values('Eintreffzeit').rename(
    columns={'BEZEICHNUN': 'Statistischer Bezirk', 'Eintreffzeit': 'Durchschnittliche Eintreffzeit'}
)

wohnbezirke_gdf[['NAME', 'Eintreffzeit']].sort_values('Eintreffzeit').rename(
    columns={'NAME': 'Wohnbezirk', 'Eintreffzeit': 'Durchschnittliche Eintreffzeit'}
)

def plot_choropleth(gdf, name_column, value_column, output_path, min_value, max_value, title="Map", cmap="YlOrRd"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Reproject GeoDataFrame to Web Mercator for basemap compatibility
    gdf = gdf.to_crs(epsg=3857)
    
    # Plot the choropleth using the specified value column
    gdf.plot(
        column=value_column,
        ax=ax,
        cmap=cmap,
        edgecolor='black',
        linewidth=0.5,
        vmin=min_value,
        vmax=max_value,
        alpha=0.6
    )
    
    # Add OpenStreetMap basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    ax.set_axis_off()
    ax.set_title(title, fontsize=15)
    
    # Create a colorbar with the same scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=min_value, vmax=max_value))
    sm._A = []  # dummy array for the scalar mappable
    cbar = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label(title, fontsize=12)
    
    plt.savefig(output_path,dpi=300,     pil_kwargs={  
        "quality":    60,
        "optimize":   True,
        "progressive": True
    }, bbox_inches='tight')
    plt.close(fig)

# Determine overall min and max Eintreffzeit across all district levels
min_value = min(
    stadtbezirke_gdf["Eintreffzeit"].min(),
    statistische_bezirke_gdf["Eintreffzeit"].min(),
    wohnbezirke_gdf["Eintreffzeit"].min()
)
max_value = max(
    stadtbezirke_gdf["Eintreffzeit"].max(),
    statistische_bezirke_gdf["Eintreffzeit"].max(),
    wohnbezirke_gdf["Eintreffzeit"].max()
)
print("Overall min:", min_value)
print("Overall max:", max_value)

# Plot average Eintreffzeit maps for each district level (saved as PNG)
# plot_choropleth(
#    stadtbezirke_gdf,
#    name_column="BEZEICHNUN",
#    value_column="Eintreffzeit",
#    output_path="../reports/figures/Stadtbezirke_Hagen_Eintreffzeit.png",
#    min_value=min_value,
#    max_value=max_value,
#    title="Durchschnittliche Hilfsfrist [min]"
#)

#plot_choropleth(
#    statistische_bezirke_gdf,
#    name_column="BEZEICHNUN",
#    value_column="Eintreffzeit",
#    output_path="../reports/figures/Statistische_Bezirke_Hagen_Eintreffzeit.png",
#    min_value=min_value,
#    max_value=max_value,
#    title="Durchschnittliche Hilfsfrist [min]"
#)

plot_choropleth(
    wohnbezirke_gdf,
    name_column="NAME",
    value_column="Eintreffzeit",
    output_path="../reports/figures/1.01-Wohnbezirke_Hagen_Eintreffzeit.jpg",
    min_value=min_value,
    max_value=max_value,
    title="Durchschnittliche Hilfsfrist [min]"
)

# Function to calculate the percentage of incidents with Eintreffzeit <= 8 and <= 12 minutes per district
def calculate_percentage_eintreffzeit(district_path, points_gdf, name_column):
    districts_gdf = gpd.read_file(district_path)
    if districts_gdf.crs != "EPSG:4326":
        districts_gdf = districts_gdf.to_crs(epsg=4326)
    
    # Spatial join
    joined_gdf = gpd.sjoin(points_gdf, districts_gdf, how="inner", predicate="within")
    
    # Group sizes
    group_sizes = joined_gdf.groupby(name_column).size()
    
    # Percentage for Eintreffzeit <= 8
    percentage_leq_8 = (joined_gdf[joined_gdf["Eintreffzeit"] <= 8]
                        .groupby(name_column)
                        .size() / group_sizes * 100)
    
    # Percentage for Eintreffzeit <= 12
    percentage_leq_12 = (joined_gdf[joined_gdf["Eintreffzeit"] <= 12]
                         .groupby(name_column)
                         .size() / group_sizes * 100)
    
    # Combine percentages into a dataframe
    percentages_df = pd.concat([percentage_leq_8, percentage_leq_12], axis=1)
    percentages_df.columns = ["Percentage_Eintreffzeit_leq_8", "Percentage_Eintreffzeit_leq_12"]
    percentages_df = percentages_df.reset_index()
    
    # Merge back to districts_gdf
    districts_gdf = districts_gdf.merge(percentages_df, on=name_column, how="left")
    districts_gdf[["Percentage_Eintreffzeit_leq_8", "Percentage_Eintreffzeit_leq_12"]] = districts_gdf[["Percentage_Eintreffzeit_leq_8", "Percentage_Eintreffzeit_leq_12"]].fillna(0)
    
    return districts_gdf

# Calculate percentage maps for each district level
stadtbezirke_pct_gdf = calculate_percentage_eintreffzeit(stadtbezirke_path, points_gdf, "BEZEICHNUN")
statistische_bezirke_pct_gdf = calculate_percentage_eintreffzeit(statistische_bezirke_path, points_gdf, "BEZEICHNUN")
wohnbezirke_pct_gdf = calculate_percentage_eintreffzeit(wohnbezirke_path, points_gdf, "NAME")

# Define min and max for the percentage scale (0 to 100)
min_percentage, max_percentage = 0, 100

stadtbezirke_pct_gdf[['BEZEICHNUN', 'Percentage_Eintreffzeit_leq_8']].sort_values('Percentage_Eintreffzeit_leq_8').rename(
    columns={'BEZEICHNUN': 'Stadtbezirk', 'Percentage_Eintreffzeit_leq_8': 'Anteil Eintreffzeit ≤ 8 min'}
)

stadtbezirke_pct_gdf[['BEZEICHNUN', 'Percentage_Eintreffzeit_leq_12']].sort_values('Percentage_Eintreffzeit_leq_12').rename(
    columns={'BEZEICHNUN': 'Stadtbezirk', 'Percentage_Eintreffzeit_leq_12': 'Anteil Eintreffzeit ≤ 12 min'}
)

statistische_bezirke_pct_gdf[['BEZEICHNUN', 'Percentage_Eintreffzeit_leq_8']].sort_values('Percentage_Eintreffzeit_leq_8').rename(
    columns={'BEZEICHNUN': 'Statistischer Bezirk', 'Percentage_Eintreffzeit_leq_8': 'Anteil Eintreffzeit ≤ 8 min'}
)

statistische_bezirke_pct_gdf[['BEZEICHNUN', 'Percentage_Eintreffzeit_leq_12']].sort_values('Percentage_Eintreffzeit_leq_12').rename(
    columns={'BEZEICHNUN': 'Statistischer Bezirk', 'Percentage_Eintreffzeit_leq_12': 'Anteil Eintreffzeit ≤ 12 min'}
)

wohnbezirke_pct_gdf[['NAME', 'Percentage_Eintreffzeit_leq_8']].sort_values('Percentage_Eintreffzeit_leq_8').rename(
    columns={'NAME': 'Wohnbezirk', 'Percentage_Eintreffzeit_leq_8': 'Anteil Eintreffzeit ≤ 8 min'}
)

# 1. Merge beider DataFrames über 'NAME'
merged = pd.merge(
    wohnbezirke_pct_gdf[['NAME', 'Percentage_Eintreffzeit_leq_12', 'Percentage_Eintreffzeit_leq_8']],
    wohnbezirke_gdf[['NAME', 'Eintreffzeit']],
    on='NAME'
)

# 2. Spalten umbenennen
merged = merged.rename(
    columns={
        'NAME': 'Wohnbezirk',
        'Eintreffzeit': 'Durchschnittliche Hilfsfrist',
        'Percentage_Eintreffzeit_leq_12': 'Anteil Hilfsfrist ≤ 12 min',
        'Percentage_Eintreffzeit_leq_8': 'Anteil Hilfsfrist ≤ 8 min'

    }
)

# 3. Sortieren und Index zurücksetzen
merged = merged.sort_values('Wohnbezirk').reset_index(drop=True)

# 4. Numerische Spalten auf 1 Nachkommastelle runden
numeric_cols = [
    'Anteil Hilfsfrist ≤ 12 min',
    'Anteil Hilfsfrist ≤ 8 min',
    'Durchschnittliche Hilfsfrist'
]
merged[numeric_cols] = merged[numeric_cols].round(1)

# LaTeX-Tabelle aufbauen
lines = [
    r"\begin{table}",
    r"\centering",
    r"\caption{Integrierte Übersicht der Hilfsfristen in den Wohnbezirken}",
    r"\label{tab:wohnbezirke_integriert}",
    r"\begin{tabular}{lrrr}",
    r"\toprule",
    r"Wohnbezirk & Durchschnittliche Hilfsfrist & Anteil Hilfsfrist $\leq$ 8 min (\%) & Anteil Hilfsfrist $\leq$ 12 min (\%)\\",
    r"\midrule",
]

for _, row in merged.iterrows():
    wb   = row['Wohnbezirk']
    dh   = row['Durchschnittliche Hilfsfrist']
    a8   = row['Anteil Hilfsfrist ≤ 8 min']
    a12  = row['Anteil Hilfsfrist ≤ 12 min']
    dh_s = f"{dh:.1f}"   if pd.notnull(dh)  else "NaN"
    a8_s = f"{a8:.1f}"   if pd.notnull(a8)  else "NaN"
    a12_s= f"{a12:.1f}"  if pd.notnull(a12) else "NaN"
    lines.append(f"{wb} & {dh_s} & {a8_s} & {a12_s}\\\\")
    
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]

# Ausgabe der LaTeX-Tabelle
print("\n".join(lines))

# Plot percentage maps (saved as PNG)
# Prozentualer Anteil der Einätze in den Stadtbezirken mit Eintreffzeit ≤ 8 Minuten

#plot_choropleth(
#    stadtbezirke_pct_gdf,
#    name_column="BEZEICHNUN",
#    value_column="Percentage_Eintreffzeit_leq_8",
#    output_path="../reports/figures/Stadtbezirke_Hagen_Percentage_Eintreffzeit_leq_8.png",
#    min_value=min_percentage,
#    max_value=max_percentage,
#    title="Anteil Hilfsfrist ≤ 8 min",
#    cmap="YlOrRd_r"
#)

# Prozentualer Anteil der Einätze in den statistischen Bezirken mit Eintreffzeit ≤ 8 Minuten
#plot_choropleth(
#    statistische_bezirke_pct_gdf,
#    name_column="BEZEICHNUN",
#    value_column="Percentage_Eintreffzeit_leq_8",
#    output_path="../reports/figures/Statistische_Bezirke_Hagen_Percentage_Eintreffzeit_leq_8.png",
#    min_value=min_percentage,
#    max_value=max_percentage,
#    title="Anteil Hilfsfrist ≤ 8 min",
#    cmap="YlOrRd_r"
#)

# Prozentualer Anteil der Einätze in den Wohnbezirken mit Eintreffzeit ≤ 8 Minuten
plot_choropleth(
    wohnbezirke_pct_gdf,
    name_column="NAME",
    value_column="Percentage_Eintreffzeit_leq_8",
    output_path="../reports/figures/1.01-Wohnbezirke_Hagen_Percentage_Eintreffzeit_leq_8.jpg",
    min_value=min_percentage,
    max_value=max_percentage,
    title="Anteil Hilfsfrist ≤ 8 min",
    cmap="YlOrRd_r"
)

# Plot percentage maps (saved as PNG)
# Prozentualer Anteil der Einätze in den Stadtbezirken mit Eintreffzeit ≤ 12 Minuten
#plot_choropleth(
#    stadtbezirke_pct_gdf,
#    name_column="BEZEICHNUN",
#    value_column="Percentage_Eintreffzeit_leq_12",
#    output_path="../reports/figures/Stadtbezirke_Hagen_Percentage_Eintreffzeit_leq_12.png",
#    min_value=min_percentage,
#    max_value=max_percentage,
#    title="Anteil Hilfsfrist ≤ 12 min",
#    cmap="YlOrRd_r"
#)

# Prozentualer Anteil der Einätze in den statistischen Bezirken mit Eintreffzeit ≤ 12 Minuten
#plot_choropleth(
#    statistische_bezirke_pct_gdf,
#    name_column="BEZEICHNUN",
#    value_column="Percentage_Eintreffzeit_leq_12",
#    output_path="../reports/figures/Statistische_Bezirke_Hagen_Percentage_Eintreffzeit_leq_12.png",
#    min_value=min_percentage,
#    max_value=max_percentage,
#    title="Anteil Hilfsfrist ≤ 12 min",
#    cmap="YlOrRd_r"
#)

# Prozentualer Anteil der Einätze in den Wohnbezirken mit Eintreffzeit ≤ 12 Minuten
plot_choropleth(
    wohnbezirke_pct_gdf,
    name_column="NAME",
    value_column="Percentage_Eintreffzeit_leq_12",
    output_path="../reports/figures/1.01-Wohnbezirke_Hagen_Percentage_Eintreffzeit_leq_12.jpg",
    min_value=min_percentage,
    max_value=max_percentage,
    title="Anteil Hilfsfrist ≤ 12 min",
    cmap="YlOrRd_r"
)

