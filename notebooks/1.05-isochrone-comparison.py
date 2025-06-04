#!/usr/bin/env python
# coding: utf-8

import openrouteservice
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import contextily as ctx

import sys
import os

sys.path.append(os.path.dirname(os.getcwd()))

import src.ors as ors
ors.start()


# Initialize OpenRouteService client
client = openrouteservice.Client(base_url='http://localhost:8080/ors')

# Define starting locations
locations = [
    (51.359159943647825, 7.477940458612092),  # Location 1
    (51.365917783741516, 7.451786216713235)   # Location 2
]

# Define a dictionary to store isochrones for each location with designated colors
isochrones_data = {
    "Location 1": {"coordinates": locations[0], "color": "blue"},
    "Location 2": {"coordinates": locations[1], "color": "red"}
}

# Request 5-minute car isochrones for each location
for label, data in isochrones_data.items():
    isochrones = client.isochrones(
        locations=[(data["coordinates"][1], data["coordinates"][0])],  # ORS uses (longitude, latitude)
        profile='driving-car',
        range=[300]  # 5 minutes in seconds
    )

    # Extract the isochrone geometry and convert it to a GeoDataFrame
    isochrone_geometry = isochrones['features'][0]['geometry']
    isochrone_polygon = gpd.GeoDataFrame(geometry=[Polygon(isochrone_geometry['coordinates'][0])], crs="EPSG:4326")
    isochrones_data[label]["polygon"] = isochrone_polygon.to_crs(epsg=3857)

# Calculate the bounding box of all polygons combined
all_bounds = [isochrones_data[label]["polygon"].total_bounds for label in isochrones_data]
minx = min(bounds[0] for bounds in all_bounds)
miny = min(bounds[1] for bounds in all_bounds)
maxx = max(bounds[2] for bounds in all_bounds)
maxy = max(bounds[3] for bounds in all_bounds)

# Calculate the center point of the overall bounding box
center_x = (minx + maxx) / 2
center_y = (miny + maxy) / 2
main_location = Point(center_x, center_y)

# Define a common extent based on the overall bounding box with additional buffer
buffer_distance = 500  # Extra buffer in meters
buffer_extent = [minx - buffer_distance, miny - buffer_distance, maxx + buffer_distance, maxy + buffer_distance]

# Plot both isochrones on the same map
fig, ax = plt.subplots(figsize=(8, 8))

for label, data in isochrones_data.items():
    # Plot the isochrone with its designated color
    data["polygon"].plot(ax=ax, edgecolor=data["color"], facecolor=data["color"], alpha=0.3, label=label)
    gdf = gpd.GeoDataFrame(geometry=[Point(data["coordinates"][1], data["coordinates"][0])], crs="EPSG:4326").to_crs(epsg=3857)

    gdf['geometry'] = gdf.buffer(5000)
    gdf.plot(ax=ax, edgecolor="red", facecolor="red", alpha=0.0)
    
    # Plot the starting point
    gdf_point = gpd.GeoDataFrame(geometry=[Point(data["coordinates"][1], data["coordinates"][0])], crs="EPSG:4326").to_crs(epsg=3857)
    gdf_point.plot(ax=ax, color=data["color"], markersize=50)

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=14)

# Set the x and y limits to the buffered bounding box
ax.set_xlim(buffer_extent[0], buffer_extent[2])
ax.set_ylim(buffer_extent[1], buffer_extent[3])

# Set up the plot appearance
ax.set_axis_off()
ax.set_aspect('equal')

# Save and display the plot
plt.tight_layout(pad=0)
plt.savefig("../reports/figures/1.05_compare_isochrones.jpg", dpi=300,
                pil_kwargs={"quality":40, "optimize":True, "progressive":True},
                bbox_inches="tight")
plt.show()

import openrouteservice
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import contextily as ctx

# Initialize OpenRouteService client
client = openrouteservice.Client(base_url='http://localhost:8080/ors')

# Define start locations
start_coords = [
    (7.462957640309114, 51.35746486686668),
    (7.546339893752056, 51.3744605710539),
    (7.4348827970092595, 51.38544480707952),
    (7.432389936920493, 51.35154303203356),
    (7.532746299850275, 51.3057632700336)
]

# Function to get combined isochrone for a given range (in seconds)
def get_combined_isochrone(client, start_coords, time_range):
    isochrones = client.isochrones(
        locations=start_coords,
        profile='driving-car',
        range=[time_range],
        location_type='start'
    )

    # Collect individual isochrone polygons
    isochrone_polygons = []
    for feature in isochrones['features']:
        isochrone_geometry = feature['geometry']
        polygon = Polygon(isochrone_geometry['coordinates'][0])
        isochrone_polygons.append(polygon)

    # Combine all polygons into a single merged polygon
    return unary_union(isochrone_polygons)

# Get combined isochrones for 12 minutes and 8 minutes
combined_isochrone_12min = get_combined_isochrone(client, start_coords, 720)  # 12 minutes
combined_isochrone_8min = get_combined_isochrone(client, start_coords, 480)   # 8 minutes

# Convert to GeoDataFrames
gdf_12min = gpd.GeoDataFrame(geometry=[combined_isochrone_12min], crs="EPSG:4326").to_crs(epsg=3857)
gdf_8min = gpd.GeoDataFrame(geometry=[combined_isochrone_8min], crs="EPSG:4326").to_crs(epsg=3857)

# Define a buffer extent for the plot area
buffer_extent = gdf_12min.total_bounds
buffer_distance = 1000  # 1 km buffer for visibility
buffer_extent_expanded = [
    buffer_extent[0] - buffer_distance,
    buffer_extent[1] - buffer_distance,
    buffer_extent[2] + buffer_distance,
    buffer_extent[3] + buffer_distance
]

# Plot both isochrones on the same map
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the 12-minute isochrone first (background layer)
gdf_12min.plot(ax=ax, edgecolor="blue", facecolor="blue", alpha=0.2, label="12-Minute Isochrone")

# Plot the 8-minute isochrone on top
gdf_8min.plot(ax=ax, edgecolor="red", facecolor="red", alpha=0.3, label="8-Minute Isochrone")

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=12)

# Plot each starting point
gdf_points = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lon, lat in start_coords], crs="EPSG:4326").to_crs(epsg=3857)
gdf_points.plot(ax=ax, color="black", markersize=50, label="Rettungswachen")

# Set up the plot limits
ax.set_xlim(buffer_extent_expanded[0], buffer_extent_expanded[2])
ax.set_ylim(buffer_extent_expanded[1], buffer_extent_expanded[3])

# Configure plot appearance
ax.set_axis_off()
ax.set_aspect('equal')
ax.legend( loc="lower right", frameon=True)

# Save and display the plot
plt.tight_layout(pad=0)
plt.savefig("../reports/figures/1.05_combined_8_12_minute_isochrone.jpg", dpi=300,
                pil_kwargs={"quality":40, "optimize":True, "progressive":True},
                bbox_inches="tight")
#plt.savefig("../reports/figures/combined_8_12_minute_isochrone.pdf", dpi=300, bbox_inches='tight')
plt.show()
