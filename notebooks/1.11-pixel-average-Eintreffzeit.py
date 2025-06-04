#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import contextily as ctx
from shapely.geometry import box

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

# ─── 2) Read point data & bin it ──────────────────────────────────────────
final_dataset_path = "../data/interim/selected-data.parquet"
bin_size = 0.001

# points
points_df = pd.read_parquet(final_dataset_path)
points_df['lat_bin'] = np.floor(points_df.EINSATZORT_lat  / bin_size) * bin_size
points_df['lon_bin'] = np.floor(points_df.EINSATZORT_lon  / bin_size) * bin_size

# average Eintreffzeit per bin
bin_avg = (
    points_df
    .groupby(['lat_bin','lon_bin'])['Eintreffzeit']
    .mean()
    .reset_index()
)
bin_avg['geometry'] = bin_avg.apply(
    lambda r: box(r.lon_bin, r.lat_bin, r.lon_bin+bin_size, r.lat_bin+bin_size),
    axis=1
)
bins_avg_gdf = gpd.GeoDataFrame(bin_avg, geometry='geometry', crs="EPSG:4326")

# percentage ≤ threshold per bin (generic function)
def make_percentage_gdf(threshold):
    grp = points_df.groupby(['lat_bin','lon_bin'])
    pct = (
        grp.apply(lambda df: (df.Eintreffzeit <= threshold).sum() / len(df) * 100)
           .reset_index(name='percentage')
    )
    pct['geometry'] = pct.apply(
        lambda r: box(r.lon_bin, r.lat_bin, r.lon_bin+bin_size, r.lat_bin+bin_size),
        axis=1
    )
    return gpd.GeoDataFrame(pct, geometry='geometry', crs="EPSG:4326")

bins_pct8_gdf  = make_percentage_gdf(8)
bins_pct12_gdf = make_percentage_gdf(12)

# classification (green if ≥90%, red otherwise)
def classify(gdf, pct_col):
    gdf['color'] = np.where(gdf[pct_col]>=90, 'green', 'red')
    return gdf

bins_clf8  = classify(bins_pct8_gdf.copy(),  'percentage')
bins_clf12 = classify(bins_pct12_gdf.copy(), 'percentage')

# ─── 3) Plotting functions with unified styling ─────────────────────────────────

def plot_bin_choropleth(gdf, value_col, title, outpath, cmap='YlOrRd'):
    fig, ax = plt.subplots(figsize=(8, 8))
    g = gdf.to_crs(epsg=3857)
    g.plot(
        column=value_col, ax=ax,
        cmap=cmap, linewidth=0, alpha=0.6,
        vmin=gdf[value_col].min(), vmax=gdf[value_col].max()
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, interpolation="bilinear")
    # overlay the city boundary
    city_outline.plot(
        ax=ax,
        edgecolor="grey",
        linewidth=1.2,
        facecolor="none",
        zorder=6
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, pad=12)
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=gdf[value_col].min(), vmax=gdf[value_col].max())
    )
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label(title, fontsize=12)
    plt.savefig(outpath,dpi=300,     pil_kwargs={  
        "quality":    20,
        "optimize":   True,
        "progressive": True
    }, bbox_inches='tight')
    plt.close(fig)

def plot_binary_classification(gdf, title, outpath):
    fig, ax = plt.subplots(figsize=(8, 8))
    g = gdf.to_crs(epsg=3857)
    for color, label in [('red', '< 90%'), ('green', '≥ 90%')]:
        subset = g[g['color'] == color]
        if not subset.empty:
            subset.plot(
                ax=ax, color=color, linewidth=0,
                alpha=0.6, label=label
            )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, interpolation="bilinear")
    # overlay the city boundary
    city_outline.plot(
        ax=ax,
        edgecolor="grey",
        linewidth=1.2,
        facecolor="none",
        zorder=6
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, pad=12)
    #plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.savefig(outpath,dpi=300,     pil_kwargs={  
        "quality":    60,
        "optimize":   True,
        "progressive": True
    }, bbox_inches='tight')

    plt.close(fig)

# ─── 4) Generate figures ───────────────────────────────────────────────────

# 4.1 average Eintreffzeit
plot_bin_choropleth(
    bins_avg_gdf,
    'Eintreffzeit',
    "Durchschnittliche Hilfsfrist [min]",
    "../reports/figures/1.11-Binned_Average_Eintreffzeit.jpg"
)

# 4.2 percentage ≤ 8 min
plot_bin_choropleth(
    bins_pct8_gdf,
    'percentage',
    "Anteil Hilfsfrist ≤ 8 min [%]",
    "../reports/figures/1.11-Binned_Percentage_Eintreffzeit_LE8.jpg",
    cmap="YlOrRd_r"
)

# 4.3 binary classification 8 min
plot_binary_classification(
    bins_clf8,
    "8 min Hilfsfrist zu 90% erreicht",
    "../reports/figures/1.11-Binned_Binary_Classification_8min.jpg"
)

# 4.4 percentage ≤ 12 min
plot_bin_choropleth(
    bins_pct12_gdf,
    'percentage',
    "Anteil Hilfsfrist ≤ 12 min [%]",
    "../reports/figures/1.11-Binned_Percentage_Eintreffzeit_LE12.jpg",
    cmap="YlOrRd_r"
)

# 4.5 binary classification 12 min
plot_binary_classification(
    bins_clf12,
    "12 min Hilfsfrist zu 90% erreicht",
    "../reports/figures/1.11-Binned_Binary_Classification_12min.jpg"
)
