#!/usr/bin/env python
# coding: utf-8

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

# 3) Prepare transformer 
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def plot_with_contextily(ds, outprefix, sigma=3, zoom=False):
    """
    Plot isochrone contours (8 min & 12 min) on a colored OSM basemap,
    with a legend instead of inline labels.
    """
    # Smooth data
    data = ds["mean_prediction"].fillna(99).values
    sm = gaussian_filter(data, sigma=sigma)

    # Build Mercator mesh
    lon2d, lat2d = np.meshgrid(ds["x"], ds["y"])
    x2d, y2d = transformer.transform(lon2d, lat2d)

    fig, ax = plt.subplots(figsize=(8, 8))
    
    city_outline.plot(
        ax=ax,
        edgecolor="grey",
        linewidth=1.2,
        facecolor="none",
        zorder=6
    )

    # Draw contour lines
    levels = [8, 12]
    contour = ax.contour(
        x2d, y2d, sm,
        levels=levels,
        colors=["red", "blue"],
        linewidths=2,
        zorder=12
    )

    # Create custom legend handles
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="red",   lw=2, label="8 Minuten"),
        Line2D([0], [0], color="blue",  lw=2, label="12 Minuten"),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        title="Isochronen",
        frameon=True
    )

    # Set extent
    if zoom:
        dx = (xmax - xmin) / 3
        dy = (ymax - ymin) / 3
        ax.set_xlim(xmin + dx, xmax - dx)
        ax.set_ylim(ymin + dy, ymax - dy)
        suffix = "zoom"
    else:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        suffix = "full"

    # Add colored OSM basemap
    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik,
        interpolation="bilinear"
    )

    ax.set_axis_off()
    ax.set_title("Hilfsfristen Isochronen\nReale Disposition", fontsize=15)

    # Save
    #plt.savefig(f"{outprefix}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{outprefix}_{suffix}.jpg",dpi=300,     pil_kwargs={  
        "quality":    40,
        "optimize":   True,
        "progressive": True
    }, bbox_inches='tight')
    plt.close(fig)

# 4) Loop over all NetCDFs
netcdf_files = [ 
    "predictions_base.nc",
]

for nc in netcdf_files:
    ds = xr.open_dataset(f"../data/processed/{nc}")
    out = f"../reports/figures/3.03_contour_{nc.replace('.nc','')}"
    plot_with_contextily(ds, out, sigma=2, zoom=False)
    #plot_with_contextily(ds, out, sigma=2, zoom=True)

# In[ ]:

