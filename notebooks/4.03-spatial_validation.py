#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import joblib
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import contextily as ctx
from shapely.geometry import box

# ─── 1) Load data & model, compute residuals ─────────────────────────────────
print("[LOG] Loading data and model...")
df = pd.read_parquet("../data/interim/selected-data_2024.parquet")
pipeline = joblib.load("../data/interim/best_RF-Eintreffzeit.pkl")

feature_columns = [
    "Standort FZ",
           "EINSATZORT_lat",
           "EINSATZORT_lon",
           "Wochentag",
           "Feiertag",
           "Ferien",
           "Monat",
           "Uhrzeit",
           "distance",
           "duration",
           "temperature_celsius",
           "snow_cover",
           "dewpoint_temperature",
           "hourly_precipitation",
           "hourly_snowfall"
]

X = df[feature_columns]
y = df["Eintreffzeit"]
y_pred = pipeline.predict(X)
df["residual"] = y - y_pred

# ─── 2) Prepare city-boundary & extent (Web-Mercator) ───────────────────────────
stadtteile = (
    gpd.read_file("../data/raw/Stadtbezirke_Hagen.shp")
       .to_crs(epsg=3857)
)
fig, ax = plt.subplots(figsize=(8, 8))
stadtteile.plot(ax=ax, edgecolor="none", alpha=0.0)
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
plt.close(fig)
city_outline = stadtteile.dissolve().boundary

# ─── 3) Bin points, count & compute mean residual per bin ───────────────────────
bin_size = 0.005

df['lat_bin'] = np.floor(df.EINSATZORT_lat / bin_size) * bin_size
df['lon_bin'] = np.floor(df.EINSATZORT_lon / bin_size) * bin_size

# Group, count and mean‐aggregate
grp = df.groupby(['lat_bin','lon_bin'])
agg = grp['residual'].agg(['mean','count']).reset_index().rename(
    columns={'mean':'mean_residual','count':'n_points'}
)

# Filter out bins with fewer than 5 points
agg = agg[agg.n_points >= 5]

agg['geometry'] = agg.apply(
    lambda r: box(r.lon_bin, r.lat_bin, r.lon_bin+bin_size, r.lat_bin+bin_size),
    axis=1
)
bins_res_gdf = gpd.GeoDataFrame(agg, geometry='geometry', crs="EPSG:4326")

# ─── 4) Plotting function for residuals ────────────────────────────────────────
def plot_bin_residuals(gdf, val_col, title, outpath, cmap='RdBu'):
    g = gdf.to_crs(epsg=3857)
    vmin, vmax = g[val_col].min(), g[val_col].max()
    norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 8))
    g.plot(column=val_col, ax=ax, cmap=cmap, norm=norm,
           linewidth=0, alpha=0.9)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, interpolation="bilinear")
    city_outline.plot(ax=ax, edgecolor="grey", linewidth=1.2,
                      facecolor="none", zorder=6)
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, pad=12)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label(title, fontsize=12)

    plt.savefig(outpath, dpi=300, pil_kwargs={
                      "quality":60, "optimize":True, "progressive":True},
                bbox_inches='tight')
    plt.close(fig)

# ─── 5) Generate & save the filtered residuals map ─────────────────────────────
plot_bin_residuals(
    bins_res_gdf,
    'mean_residual',
    "Durchschnittliches Residuum [Minuten]",
    "../reports/figures/4.03-Binned_Average_Validation.jpg"
)
