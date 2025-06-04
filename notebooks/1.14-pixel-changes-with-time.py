#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import contextily as ctx
from scipy.stats import linregress
from shapely.geometry import box

# ─── 1) Load city districts and get Web-Mercator bounds ───────────────
stadtteile = (
    gpd.read_file("../data/raw/Stadtbezirke_Hagen.shp")
       .to_crs(epsg=3857)
)
# Use an invisible plot to capture bounds
fig, ax = plt.subplots(figsize=(8, 8))
stadtteile.plot(ax=ax, alpha=0)
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
plt.close(fig)

# Dissolve to get city outline
city_outline = stadtteile.dissolve().boundary

# ─── 2) Read point data and project to Web-Mercator ─────────────────
points = pd.read_parquet("../data/interim/selected-data.parquet")
points = gpd.GeoDataFrame(
    points,
    geometry=gpd.points_from_xy(points.EINSATZORT_lon, points.EINSATZORT_lat),
    crs="EPSG:4326"
).to_crs(epsg=3857)
points["Hilfsfrist"] = points.Eintreffzeit  # or rename column if needed

# ─── 3) Create square grid (bins) ────────────────────────────────────
bin_size = 500  # in meters (adjust for resolution)
x_bins = np.arange(xmin, xmax + bin_size, bin_size)
y_bins = np.arange(ymin, ymax + bin_size, bin_size)

# Build grid index list
grid = []
for x0 in x_bins[:-1]:
    for y0 in y_bins[:-1]:
        grid.append(box(x0, y0, x0 + bin_size, y0 + bin_size))

grid_gdf = gpd.GeoDataFrame(
    {'geometry': grid},
    crs="EPSG:3857"
)

# Spatial join points to grid
joined = gpd.sjoin(points, grid_gdf.reset_index().rename(columns={'index':'cell_id'}), how="inner", predicate="within")

# Compute counts per cell and year
counts = joined.groupby(['cell_id', 'Jahr']).size().reset_index(name='count')
# Identify cells with >=10 datapoints in every year they appear
good_cells = counts.groupby('cell_id')['count'].min().loc[lambda x: x >= 5].index
# Keep only valid cell-year records
joined = joined[joined['cell_id'].isin(good_cells)]

# ─── 4) Compute per-year average Hilfsfrist per cell ────────────────
cell_yearly = (
    joined
    .groupby(['cell_id', 'Jahr'])['Hilfsfrist']
    .mean()
    .reset_index()
)

# ─── 5) Compute regression slope per cell ────────────────────────────
records = []
for cell_id, group in cell_yearly.groupby('cell_id'):
    group = group.sort_values('Jahr')
    x = group['Jahr'].values
    y = group['Hilfsfrist'].values
    if len(x) >= 2:
        slope, intercept, r_val, p_val, std_err = linregress(x, y)
    else:
        slope, p_val = np.nan, np.nan
    records.append({'cell_id': cell_id, 'slope': slope, 'p_value': p_val})

slope_df = pd.DataFrame(records)

# ─── 6) Merge slopes back to grid and plot ────────────────────────────
grid_plot = grid_gdf.reset_index().rename(columns={'index':'cell_id'}).merge(
    slope_df, on='cell_id', how='left'
)

# In[ ]:

from matplotlib.colors import TwoSlopeNorm
# Plot function for slope choropleth
def plot_slope_map(gdf, column, title, outpath, cmap='coolwarm'):
    
    min_slope = gdf["slope"].min()
    max_slope = gdf["slope"].max()
    abs_max = max(abs(min_slope), abs(max_slope))

    # create a norm that centers at 0
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    
    g = gdf.to_crs(epsg=3857)
    vmin, vmax = np.nanpercentile(gdf[column], [5, 95])
    fig, ax = plt.subplots(figsize=(8, 8))
    g = gdf.to_crs(epsg=3857)
    g.plot(
        column="slope",
        ax=ax,
        cmap="coolwarm",
        norm=norm,               
        linewidth=0.0,
        alpha=0.9
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    city_outline.plot(ax=ax, edgecolor='grey', linewidth=1, facecolor='none')
    ax.set_axis_off()
    ax.set_title(title,fontsize=16)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label('Jährliche Veränderung der Hilfsfrist [min/a]', fontsize=12)
    plt.savefig(outpath,dpi=300,     pil_kwargs={  
        "quality":    60,
        "optimize":   True,
        "progressive": True
    }, bbox_inches='tight')
    plt.close(fig)

# Generate the slope map
generate_path = "../reports/figures/1.14-binned_slope_hilfsfrist.jpg"
plot_slope_map(
    grid_plot,
    'slope',
    'Jährliche Veränderung der Hilfsfrist',
    generate_path
)

# In[ ]:

