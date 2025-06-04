import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import contextily as ctx
import pyproj

# Define the list of NetCDF files to process
netcdf_files = [
    "predictions_log_base.nc",
    "predictions_log_feiertag_false.nc",
    "predictions_log_feiertag_true.nc",
    "predictions_log_ferien_false.nc",
    "predictions_log_ferien_true.nc",
    "predictions_log_snowcover0.8_temp0.0.nc",
    "predictions_log_uhrzeit_0.nc",
    "predictions_log_uhrzeit_4.nc",
    "predictions_log_uhrzeit_8.nc",
    "predictions_log_uhrzeit_12.nc",
    "predictions_log_uhrzeit_16.nc",
    "predictions_log_uhrzeit_20.nc",
    "predictions_log_wochentag_saturday.nc",
    "predictions_log_wochentag_sunday.nc",
    "predictions_log_wochentag_weekdayrandom.nc",
]

# -----------------------------------------------------------------------------
# Helper function to plot a single dataset: full extent or zoomed.
# -----------------------------------------------------------------------------
def plot_with_contextily(ds, zoom=False, outprefix="output_plot"):
    """
    Plot the mean_prediction variable on top of a greyscale CartoDB basemap.
    If zoom=True, plot only the central 1/3 of the domain.
    Saves PNG and EPS files at 300 dpi.
    """
    # Extract coordinates and data
    lon = ds["x"]
    lat = ds["y"]
    data = ds["mean_prediction"]

    # Determine full extent (EPSG:4326)
    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())

    # Zoom: select the central 1/3 of the domain
    if zoom:
        lon_extent = lon_max - lon_min
        lat_extent = lat_max - lat_min
        lon_min_zoom = lon_min + lon_extent / 3
        lon_max_zoom = lon_min + 2 * lon_extent / 3
        lat_min_zoom = lat_min + lat_extent / 3
        lat_max_zoom = lat_min + 2 * lat_extent / 3

        ds = ds.sel(x=slice(lon_min_zoom, lon_max_zoom),
                    y=slice(lat_min_zoom, lat_max_zoom))
        lon = ds["x"]
        lat = ds["y"]
        data = ds["mean_prediction"]
        lon_min, lon_max = float(lon.min()), float(lon.max())
        lat_min, lat_max = float(lat.min()), float(lat.max())

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create 2D meshgrid from 1D coordinate arrays
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Transform the meshgrid coordinates from EPSG:4326 to EPSG:3857
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x2d, y2d = transformer.transform(lon2d, lat2d)

    # Plot data with pcolormesh using the transformed coordinates.
    mesh = ax.pcolormesh(x2d, y2d, data, cmap="viridis", alpha=0.5,
                          shading="auto", zorder=3)
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.7)
    cbar.set_label("Mean Prediction")

    # Set the extent in the transformed coordinates
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add the basemap; ensure it is drawn at a lower zorder

    provider = ctx.providers.Stadia.StamenTonerLite(api_key="b0cd7fb3-b785-4366-8522-4269cc2721e4")
    provider['url'] = provider["url"] + "?api_key={api_key}"
    ctx.add_basemap(ax, source=provider)

    # Set title and filename based on zoom setting
    if zoom:
        ax.set_title("Mean Prediction (Zoomed to Center 1/3)")
        filename = f"{outprefix}_zoom"
    else:
        ax.set_title("Mean Prediction (Full Extent)")
        filename = f"{outprefix}_full"

    # Save the plot in PNG and EPS formats at 300 dpi
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



# -----------------------------------------------------------------------------
# Main Loop over all NetCDF files
# -----------------------------------------------------------------------------
for nc_file in netcdf_files:
    # Open the dataset
    ds = xr.open_dataset("../data/processed/"+nc_file)

    # Derive an output prefix from the filename (strip .nc for example)
    outprefix = "../reports/figures/"+nc_file.replace(".nc", "")

    # 1) Plot full extent
    plot_with_contextily(ds, zoom=False, outprefix=outprefix)

    # 2) Plot center 1/3
    plot_with_contextily(ds, zoom=True, outprefix=outprefix)
