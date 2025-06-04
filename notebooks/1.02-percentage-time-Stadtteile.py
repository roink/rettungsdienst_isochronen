#!/usr/bin/env python
# coding: utf-8

import geopandas as gpd
import pandas as pd
import folium
import sys
import os

sys.path.insert(0, os.path.abspath("../src"))
from plotting import plot_shapefile_with_labels

# Paths
stadtbezirke_path = "../data/raw/Stadtbezirke_Hagen.shp"
statistische_bezirke_path = "../data/raw/Statistische_Bezirke_Hagen.shp"
wohnbezirke_path = "../data/raw/Wohnbezirke_Hagen.shp"
points_path = "../data/interim/toy_data.csv"

# Load points data
points_df = pd.read_csv(points_path)
points_gdf = gpd.GeoDataFrame(
    points_df, geometry=gpd.points_from_xy(points_df.lon, points_df.lat), crs="EPSG:4326"
)

def plot_colored_map(gdf, map_location, name_column, value_column, output_path, min_value, max_value):
    # Create the base map
    m = folium.Map(location=map_location, zoom_start=12)

    # Define a consistent color scale with at least 3 values
    threshold_scale = [
        min_value,
        min_value + (max_value - min_value) * 0.25,
        (min_value + max_value) / 2,
        min_value + (max_value - min_value) * 0.75,
        max_value
    ]

    # Add the color-coded layer with the updated threshold scale
    folium.Choropleth(
        geo_data=gdf,
        data=gdf,
        columns=[name_column, value_column],
        key_on=f"feature.properties.{name_column}",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"Average Anfahrtszeit ({value_column})",
        threshold_scale=threshold_scale
    ).add_to(m)
    
    # Save and return the map
    m.save(output_path)
    return m

def calculate_center_from_shapefile(shapefile_path: str):
    # Load the shapefile using geopandas
    gdf = gpd.read_file(shapefile_path)
    
    # Reproject to WGS84 if necessary
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    
    # Get the bounding box (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = gdf.total_bounds
    print(minx, miny, maxx, maxy)
    # Calculate the center coordinates
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    
    return center_y, center_x  # Return in folium's expected (lat, lon) order

# Plot maps
map_location = calculate_center_from_shapefile(stadtbezirke_path)

def calculate_percentage_anfahrtszeit_leq_8(district_path, points_gdf, name_column):
    # Load district shapefile
    districts_gdf = gpd.read_file(district_path)
    if districts_gdf.crs != "EPSG:4326":
        districts_gdf = districts_gdf.to_crs(epsg=4326)

    # Spatial join points with district polygons
    joined_gdf = gpd.sjoin(points_gdf, districts_gdf, how="inner", predicate="within")

    # Calculate the percentage of entries with Anfahrtszeit <= 8 for each district
    percentage_leq_8 = joined_gdf[joined_gdf["Anfahrtszeit"] > 8].groupby(name_column).size() / joined_gdf.groupby(name_column).size() * 100
    percentage_leq_8 = percentage_leq_8.reset_index().rename(columns={0: "Percentage_Anfahrtszeit_leq_8"})
    
    # Merge this percentage data back to the districts GeoDataFrame
    districts_gdf = districts_gdf.merge(percentage_leq_8, on=name_column, how="left")
    districts_gdf["Percentage_Anfahrtszeit_leq_8"] = districts_gdf["Percentage_Anfahrtszeit_leq_8"].fillna(0)  # Fill any NaN with 0%
    
    return districts_gdf

# Calculate percentages for each district level
stadtbezirke_gdf = calculate_percentage_anfahrtszeit_leq_8(stadtbezirke_path, points_gdf, "BEZEICHNUN")
statistische_bezirke_gdf = calculate_percentage_anfahrtszeit_leq_8(statistische_bezirke_path, points_gdf, "BEZEICHNUN")
wohnbezirke_gdf = calculate_percentage_anfahrtszeit_leq_8(wohnbezirke_path, points_gdf, "NAME")

# Define min and max for consistent color scaling (0 to 100 for percentage)
min_percentage, max_percentage = 0, 100

# Plot each map
map_location = calculate_center_from_shapefile(stadtbezirke_path)

_ =plot_colored_map(stadtbezirke_gdf, map_location, "BEZEICHNUN", "Percentage_Anfahrtszeit_leq_8", "../reports/figures/Stadtbezirke_Hagen_Percentage_Anfahrtszeit_leq_8.html", min_percentage, max_percentage)

_ = plot_colored_map(statistische_bezirke_gdf, map_location, "BEZEICHNUN", "Percentage_Anfahrtszeit_leq_8", "../reports/figures/Statistische_Bezirke_Hagen_Percentage_Anfahrtszeit_leq_8.html", min_percentage, max_percentage)

_ = plot_colored_map(wohnbezirke_gdf, map_location, "NAME", "Percentage_Anfahrtszeit_leq_8", "../reports/figures/Wohnbezirke_Hagen_Percentage_Anfahrtszeit_leq_8.html", min_percentage, max_percentage)

# In[ ]:

