#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import pyproj
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D

# 1) Load & reproject Wohnbezirke to Web Mercator, grab its bounds:
districts = (
    gpd.read_file("../data/raw/Wohnbezirke_Hagen.shp")
       .to_crs(epsg=3857)
)
fig, ax = plt.subplots()
districts.plot(ax=ax)
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
plt.close(fig)

# 2) Prepare lon/lat → WebMercator transformer:
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# 3) Load probability datasets:
ds8  = xr.open_dataset("../data/processed/prob8min_base.nc")
ds12 = xr.open_dataset("../data/processed/prob12min_base.nc")

# 4) Variable name in the NetCDF (adjust if different):
var = "mean_prediction"

# 5) Build Mercator mesh once:
lon2d, lat2d = np.meshgrid(ds8["x"], ds8["y"])
x2d, y2d = transformer.transform(lon2d, lat2d)

# 6) Set smoothing parameter:
sigma = 2  # adjust as desired

# 7) Prepare smoothed arrays:
arr8  = ds8[var].fillna(0).values
arr12 = ds12[var].fillna(0).values

sm8  = gaussian_filter(arr8,  sigma=sigma)
sm12 = gaussian_filter(arr12, sigma=sigma)

# 8) Plot both 0.9‐level contours on one map:
fig, ax = plt.subplots(figsize=(8, 8))

# 8 min isochrone (P = 0.9)
ax.contour(
    x2d, y2d, sm8,
    levels=[0.9],
    colors=["red"],
    linewidths=2,
    zorder=4
)

# 12 min isochrone (P = 0.9)
ax.contour(
    x2d, y2d, sm12,
    levels=[0.9],
    colors=["blue"],
    linewidths=2,
    zorder=4
)

# 9) Legend in lower right:
handles = [
    Line2D([0], [0], color="red",  lw=2, label="8 Min"),
    Line2D([0], [0], color="blue", lw=2, label="12 Min")
]
ax.legend(handles=handles, loc="lower right", frameon=True, title="")

# 10) Fix extent to Wohnbezirke bounds:
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# 11) Add colored OSM basemap:
ctx.add_basemap(
    ax,
    source=ctx.providers.OpenStreetMap.Mapnik,
    interpolation="bilinear"
)

# 12) Finish styling & save:
ax.set_axis_off()
ax.set_title("90% Erreichungsgrad", fontsize=16)

plt.savefig("../reports/figures/3.04-isochrones_0.9_combined.jpg",
        dpi=300,
        pil_kwargs={  
            "quality":    40,
            "optimize":   True,
            "progressive": True
        }, 
        bbox_inches='tight')
plt.close(fig)
