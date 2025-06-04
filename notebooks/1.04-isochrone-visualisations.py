#!/usr/bin/env python
# coding: utf-8

import numpy as np
import folium
import sys
import os
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

import sys

sys.path.append(os.path.dirname(os.getcwd()))

import src.ors as ors
import openrouteservice

ors.start()

# Initialize OpenRouteService client
client = openrouteservice.Client(base_url='http://localhost:8080/ors')

# Define start location
location_wgs84 = (7.461627698166156, 51.36219042530354)  # (longitude, latitude)

# Request 5-minute car isochrone
isochrones = client.isochrones(
    locations=[location_wgs84],
    profile='driving-car',
    range=[300]  # 5 minutes in seconds
)

# Extract the isochrone geometry and convert it to a GeoDataFrame
isochrone_geometry = isochrones['features'][0]['geometry']
isochrone_polygon = gpd.GeoDataFrame(geometry=[Polygon(isochrone_geometry['coordinates'][0])], crs="EPSG:4326")

# Define main location and buffer size
location_wgs84 = Point(7.461627698166156, 51.36219042530354)
buffer_distance = 3000  # 2500 meters to give some extra space around the 2 km circle

# Create a GeoDataFrame for the main point and reproject to Web Mercator
gdf = gpd.GeoDataFrame(geometry=[location_wgs84], crs="EPSG:4326").to_crs(epsg=3857)

# Buffer the point to create an extent that includes the 2 km radius and some extra space
buffer_extent = gdf.buffer(buffer_distance).total_bounds  # [minx, miny, maxx, maxy]

# Create a 2 km buffer around the point
gdf['geometry'] = gdf.buffer(2000)

# Plot the buffered area
fig, ax = plt.subplots(figsize=(6, 6))
gdf.plot(ax=ax, edgecolor="red", facecolor="red", alpha=0.3)

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=14)

# Plot the starting point
gdf_point = gpd.GeoDataFrame(geometry=[location_wgs84], crs="EPSG:4326").to_crs(epsg=3857)
gdf_point.plot(ax=ax, color="red", markersize=50)

# Set the x and y limits to the buffer extent for consistency
ax.set_xlim(buffer_extent[0], buffer_extent[2])
ax.set_ylim(buffer_extent[1], buffer_extent[3])

# Remove axis ticks and labels, set aspect ratio to equal
ax.set_axis_off()
ax.set_aspect('equal')  # Ensures the circle isn't distorted

# Save and display the plot
plt.tight_layout(pad=0)
plt.savefig("../reports/figures/1.04_2km_circle.jpg", dpi=300,
                pil_kwargs={"quality":40, "optimize":True, "progressive":True},
                bbox_inches="tight")
plt.show()

# Reproject to Web Mercator for plotting with Contextily
isochrone_polygon = isochrone_polygon.to_crs(epsg=3857)

# Assuming `isochrone_polygon` contains the 10-minute isochrone polygon
fig, ax = plt.subplots(figsize=(6, 6))
isochrone_polygon.plot(ax=ax, edgecolor="blue", facecolor="blue", alpha=0.3)
gdf.plot(ax=ax, edgecolor="red", facecolor="red", alpha=0.0)

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=14)

# Plot the starting point
gdf_point = gpd.GeoDataFrame(geometry=[location_wgs84], crs="EPSG:4326").to_crs(epsg=3857)
gdf_point.plot(ax=ax, color="red", markersize=50)

# Set the x and y limits to the buffer extent for consistency
ax.set_xlim(buffer_extent[0], buffer_extent[2])
ax.set_ylim(buffer_extent[1], buffer_extent[3])

# Remove axis ticks and labels, set aspect ratio to equal
ax.set_axis_off()
ax.set_aspect('equal')

# Save and display the plot
plt.tight_layout(pad=0)
plt.savefig("../reports/figures/1.04_5_minute_isochrone.jpg", dpi=300,
                pil_kwargs={"quality":40, "optimize":True, "progressive":True},
                bbox_inches="tight")
plt.show()

