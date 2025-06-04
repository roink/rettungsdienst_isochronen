#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Point
from scipy.spatial import cKDTree

# Load points data
points_df = pd.read_csv("../data/interim/toy_data.csv")
x = points_df['lon'].values
y = points_df['lat'].values
z = points_df['Anfahrtszeit'].values

shapefile_path = "../data/raw/Stadtbezirke_Hagen.shp"

    # Load the shapefile using geopandas
gdf = gpd.read_file(shapefile_path)
    
    # Reproject to WGS84 if necessary
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs(epsg=4326)
    
    # Get the bounding box (minx, miny, maxx, maxy)
minx, miny, maxx, maxy = gdf.total_bounds
print(minx, miny, maxx, maxy)

center_x = (minx + maxx) / 2
center_y = (miny + maxy) / 2

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy as sp

# Define a grid for interpolation
grid_x, grid_y = np.mgrid[minx:maxx:500j, miny:maxy:500j]

grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

# Create KDTree for nearest neighbor search
points = np.vstack([x, y]).T
tree = cKDTree(points)

# Find the 5 nearest neighbors and compute the mean for each grid point
distances, indices = tree.query(grid_points, k=50)
grid_z = np.mean(z[indices], axis=1)

# Reshape to match grid shape for contour plotting
grid_z = grid_z.reshape(grid_x.shape)
sigma = [5, 5]
grid_z = sp.ndimage.filters.gaussian_filter(grid_z, sigma, mode='constant')

# Plot iso-lines
plt.figure()
contour = plt.contour(grid_x, grid_y, grid_z, levels=10, cmap='viridis')
plt.colorbar(label='Value')
#plt.scatter(x, y, c=z, edgecolor='k')  # plot original points for reference
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Iso-lines of Irregularly Spaced Data using 5-Nearest Neighbors Mean')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import folium
import branca
from folium import plugins
from scipy.interpolate import griddata
import geojsoncontour
import scipy as sp
import scipy.ndimage

# Plot iso-lines
plt.figure()
# Create the contour
contourf = plt.contourf(grid_x, grid_y, grid_z,levels=15, alpha=0.5, linestyles='None')
#plt.scatter(x, y, c=z, edgecolor='k')  # plot original points for reference
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Iso-lines of Irregularly Spaced Data using 5-Nearest Neighbors Mean')
plt.show()

# Convert matplotlib contourf to geojson
geojson = geojsoncontour.contourf_to_geojson(
    contourf=contourf,
    min_angle_deg=3.0,
    ndigits=5,
    stroke_width=1,
    fill_opacity=0.1)
 
# Set up the map placeholdder
geomap1 = folium.Map([center_y,center_x], zoom_start=12, tiles="OpenStreetMap")# Plot the contour on Folium map
folium.GeoJson(
    geojson,
    style_function=lambda x: {
        'color':     x['properties']['stroke'],
        'weight':    x['properties']['stroke-width'],
        'fillColor': x['properties']['fill'],
        'opacity':   0.5,
    }).add_to(geomap1)

# Add the legend to the map
_ = plugins.Fullscreen(position='topright', force_separate_button=True).add_to(geomap1)

# In[ ]:

