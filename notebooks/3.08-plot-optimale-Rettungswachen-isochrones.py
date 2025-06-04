#!/usr/bin/env python
# coding: utf-8

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import contextily as ctx
import pyproj
import geopandas as gpd

# ─── 1) Load & project Stadtbezirke once, grab its Web-Mercator bounds ─────────
stadtteile = (
    gpd.read_file("../data/raw/Stadtbezirke_Hagen.shp")
       .to_crs(epsg=3857)
)
fig, ax = plt.subplots(figsize=(8, 8))
stadtteile.plot(ax=ax, edgecolor="none", alpha=0.0)  # invisible, just to set limits
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
plt.close(fig)
extent = (xmin, xmax, ymin, ymax)

# ─── 1b) build a single city‐boundary GeoSeries in Web Mercator ─────────
stadtteile_3857 = stadtteile.to_crs(epsg=3857)
city_outline = stadtteile_3857.dissolve().boundary

transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def plot_with_contextily(da, outprefix, title, sigma=3, zoom=False):
    """
    Plot isochrone contours (8 min & 12 min) of DataArray `da` on an OSM basemap.
    """
    data = da.fillna(99).values
    sm = gaussian_filter(data, sigma=sigma)

    # build Mercator mesh
    lon2d, lat2d = np.meshgrid(da['x'], da['y'])
    x2d, y2d = transformer.transform(lon2d, lat2d)

    fig, ax = plt.subplots(figsize=(8, 8))
    city_outline.plot(ax=ax, edgecolor="grey", linewidth=1.2, facecolor="none", zorder=6)

    levels = [8, 12]
    contour = ax.contour(x2d, y2d, sm, levels=levels,
                         colors=["red", "blue"], linewidths=2, zorder=12)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="red",  lw=2, label="8 Minuten"),
        Line2D([0], [0], color="blue", lw=2, label="12 Minuten"),
    ]
    ax.legend(handles=handles, loc="lower right", title="Isochronen", frameon=True)

    if zoom:
        dx, dy = (xmax - xmin)/3, (ymax - ymin)/3
        ax.set_xlim(xmin+dx, xmax-dx)
        ax.set_ylim(ymin+dy, ymax-dy)
        suffix = "zoom"
    else:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        suffix = "full"

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, interpolation="bilinear")
    ax.set_axis_off()
    ax.set_title(title, fontsize=15)

    plt.savefig(f"{outprefix}_{suffix}.jpg", dpi=300,
                pil_kwargs={"quality":40, "optimize":True, "progressive":True},
                bbox_inches="tight")
    plt.close(fig)

netcdf_files = [ 
    'predictions_Rettungswache Allgemeine Krankenhaus Hagen.nc',
    'predictions_Rettungswache Dahl.nc',
    'predictions_Rettungswache Ev.Krhs. Haspe.nc',
    'predictions_Rettungswache HaTüWe.nc',
    'predictions_Rettungswache Mitte.nc',
    'predictions_Rettungswache Ost.nc',
    'predictions_Rettungswache St. Johannes Hospital.nc',
    'predictions_Rettungswache Vorhalle.nc'
]

arrays = []
for nc in netcdf_files:
    ds = xr.open_dataset(nc)
    # fill NaN with a large number so they don't become the min
    arr = ds["mean_prediction"].fillna(99).values
    arrays.append(arr)
    ds.close()

stacked = np.stack(arrays, axis=0)       # shape (8, 1000, 1000)
min_map = np.min(stacked, axis=0)       # shape (1000, 1000)

# Wrap back into a DataArray with the same coords as the last file
# (they all share identical x and y)
template = xr.open_dataset(netcdf_files[0])
da_min = xr.DataArray(
    min_map,
    coords=[template['y'], template['x']],
    dims=["y","x"],
    name="mean_prediction_min"
)
template.close()

outprefix = "../reports/figures/3.08_contour_min_all"
title = "Hilfsfristen Isochronen\nOptimale Disposition"
plot_with_contextily(da_min, outprefix, title, sigma=2, zoom=False)

# In[ ]:

