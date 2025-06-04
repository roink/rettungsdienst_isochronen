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

extent

# In[ ]:

# 2) Prepare lon/lat → WebMercator transformer:
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# 3) per-year NetCDFs:
years     = [2018, 2019, 2020, 2021, 2022, 2023]
nc_files  = [f"../data/processed/predictions_base_{y}.nc" for y in years]

# 4) Build Mercator mesh once (all files share the same grid assumption):
ds0       = xr.open_dataset(nc_files[0])
lon2d, lat2d = np.meshgrid(ds0["x"], ds0["y"])
x2d, y2d     = transformer.transform(lon2d, lat2d)

transformer.transform(51.32, 7.37)

xmin,ymin = transformer.transform(7.36,51.31)
xmax,ymax = transformer.transform(7.48,51.39)

def plot_isochrones(level, cmap_name, title, outpath):
    """
    level     : contour level in minutes (8 or 12)
    cmap_name : e.g. 'viridis', 'plasma', etc.
    title     : figure title
    outpath   : where to save
    """
    sigma=3
    # pick a colormap and sample one color per year
    cmap   = plt.get_cmap(cmap_name)
    colors = [cmap(i / (len(years) - 1)) for i in range(len(years))]

    fig, ax = plt.subplots(figsize=(8, 8))
    
    city_outline.plot(
        ax=ax,
        edgecolor="grey",
        linewidth=1.2,
        facecolor="none",
        zorder=3
    )

    for nc, yr, col in zip(nc_files, years, colors):
        ds  = xr.open_dataset(nc)
        arr = ds["mean_prediction"].fillna(99).values
        sm  = gaussian_filter(arr, sigma=sigma)

        ax.contour(
            x2d, y2d, sm,
            levels=[level],
            colors=[col],
            linewidths=2,
            linestyles='-',
            alpha=0.8,
            zorder=4
        )

    # Legend: one handle per year
    handles = [
        Line2D([0], [0], color=col, lw=2, label=str(yr))
        for col, yr in zip(colors, years)
    ]
    ax.legend(handles=handles, title="Jahr", loc="lower right", frameon=True)

    # fixed extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # colored basemap
    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik,
        interpolation="bilinear"
    )

    ax.set_axis_off()
    ax.set_title(title, fontsize=16)

    #plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.savefig(outpath,dpi=300,     pil_kwargs={  
        "quality":    40,
        "optimize":   True,
        "progressive": True
    }, bbox_inches='tight')
    plt.close(fig)

# 6) Generate the two plots:
plot_isochrones(
    level=8,
    cmap_name="coolwarm",
    title="8-Minuten Isochronen 2018–2023",
    outpath="../reports/figures/3.06-isochrones_8min_2018-23_Westen.jpg"
)

plot_isochrones(
    level=12,
    cmap_name="coolwarm",
    title="12-Minuten Isochronen 2018–2023",
    outpath="../reports/figures/3.06-isochrones_12min_2018-23_Westen.jpg"
)

# In[ ]:

