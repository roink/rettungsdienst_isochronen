#!/usr/bin/env python
# coding: utf-8

import geopandas as gpd
import folium
import sys
import os

sys.path.insert(0, os.path.abspath("../src"))
from plotting import plot_shapefile_with_labels

stadtbezirke_path = "../data/raw/Stadtbezirke_Hagen.shp"
gdf = gpd.read_file(stadtbezirke_path)
print(str(len(gdf)) + " Stadtbezirke")
gdf.head

statistische_bezirke_path = "../data/raw/Statistische_Bezirke_Hagen.shp"
gdf = gpd.read_file(statistische_bezirke_path)
print(str(len(gdf)) + " Statistische_Bezirke")
gdf.head

wohnbezirke_path = "../data/raw/Wohnbezirke_Hagen.shp"
gdf = gpd.read_file(wohnbezirke_path)
print(str(len(gdf)) + " Wohnbezirke")
gdf.head

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

map_location = calculate_center_from_shapefile(stadtbezirke_path)  # Coordinates to center the map on
map_location

stadtbezirke_map = plot_shapefile_with_labels(stadtbezirke_path, map_location,name_column="BEZEICHNUN")

# Save the map
_ =stadtbezirke_map.save("../reports/figures/Stadtbezirke_Hagen.html")

statistische_bezirke_map = plot_shapefile_with_labels(statistische_bezirke_path, map_location,name_column="BEZEICHNUN")

# Save the map
_ = statistische_bezirke_map.save("../reports/figures/Statistische_Bezirke_Hagen.html")

wohnbezirke_map = plot_shapefile_with_labels(wohnbezirke_path, map_location,name_column="NAME")

# Save the map
_ = wohnbezirke_map.save("../reports/figures/Wohnbezirke_Hagen.html")

# In[ ]:

