import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import contextily as ctx
import pyproj

# -----------------------------------------------------------------------------
# Define the list of NetCDF files (including the base case)
# -----------------------------------------------------------------------------
netcdf_files = [
    "predictions_base.nc",
    "predictions_feiertag_false.nc",
    "predictions_feiertag_true.nc",
    "predictions_ferien_false.nc",
    "predictions_ferien_true.nc",
    "predictions_snowcover0.8_temp0.0.nc",
    "predictions_uhrzeit_0.nc",
    "predictions_uhrzeit_4.nc",
    "predictions_uhrzeit_8.nc",
    "predictions_uhrzeit_12.nc",
    "predictions_uhrzeit_16.nc",
    "predictions_uhrzeit_20.nc",
    "predictions_wochentag_saturday.nc",
    "predictions_wochentag_sunday.nc",
    "predictions_wochentag_weekdayrandom.nc",
]

# Define data and output paths
data_path = "../data/processed/"
output_path = "../reports/figures/"

# -----------------------------------------------------------------------------
# Load the base dataset
# -----------------------------------------------------------------------------
base_file = "predictions_base.nc"
ds_base = xr.open_dataset(data_path + base_file)

# -----------------------------------------------------------------------------
# Helper function to plot the difference with a basemap.
# -----------------------------------------------------------------------------
def plot_difference_with_contextily(diff, coords, title, outfilename, zoom=False):
    """
    Plot the pointwise difference with a CartoDB basemap.
    
    Parameters:
      diff      : xarray.DataArray containing the difference values.
      coords    : xarray object with "x" and "y" coordinate arrays.
      title     : Title for the plot (will include the file name).
      outfilename: Output filename (without extension) for saving the plot.
      zoom      : If True, plots only the central 1/3 of the domain.
    """
    # Extract coordinates and difference data
    lon = coords["x"]
    lat = coords["y"]
    data = diff

    # Full extent in EPSG:4326
    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())

    # If zoom is True, select the central 1/3 of the domain
    if zoom:
        lon_extent = lon_max - lon_min
        lat_extent = lat_max - lat_min
        lon_min_zoom = lon_min + lon_extent / 3
        lon_max_zoom = lon_min + 2 * lon_extent / 3
        lat_min_zoom = lat_min + lat_extent / 3
        lat_max_zoom = lat_min + 2 * lat_extent / 3

        # Subset the difference data
        data = data.sel(x=slice(lon_min_zoom, lon_max_zoom),
                        y=slice(lat_min_zoom, lat_max_zoom))
        lon = data["x"]
        lat = data["y"]
        lon_min, lon_max = float(lon.min()), float(lon.max())
        lat_min, lat_max = float(lat.min()), float(lat.max())

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create a 2D meshgrid from the 1D coordinate arrays
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Transform coordinates from EPSG:4326 to EPSG:3857
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x2d, y2d = transformer.transform(lon2d, lat2d)

    # Plot the difference using pcolormesh
    mesh = ax.pcolormesh(x2d, y2d, data, cmap="coolwarm", alpha=0.5,
                          shading="auto", zorder=3)
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.7)
    cbar.set_label("Difference in Mean Prediction")

    # Set axis limits based on transformed coordinates
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add the basemap (with a specified provider and API key)
    provider = ctx.providers.Stadia.StamenTonerLite(api_key="b0cd7fb3-b785-4366-8522-4269cc2721e4")
    provider['url'] = provider["url"] + "?api_key={api_key}"
    ctx.add_basemap(ax, source=provider)

    # Set the plot title
    ax.set_title(title)

    # Save the plot as a PNG file at 300 dpi
    plt.savefig(f"{outfilename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main Loop: Process each file (skip the base case) and plot the differences.
# -----------------------------------------------------------------------------
for nc_file in netcdf_files:
    if nc_file == base_file:
        continue  # Skip the base case

    # Load the current dataset
    ds_other = xr.open_dataset(data_path + nc_file)
    
    # Compute the pointwise difference: (other file - base)
    diff = ds_other["mean_prediction"] - ds_base["mean_prediction"]

    # Define the output filename and plot title
    outfilename = output_path + "diff_" + nc_file.replace(".nc", "")
    title = f"Difference in Mean Prediction: {nc_file}"

    # Plot the full extent difference
    plot_difference_with_contextily(diff, ds_base, title, outfilename, zoom=False)
    # Optionally, plot the zoomed view (central 1/3 of the domain)
    plot_difference_with_contextily(diff, ds_base, title + " (Zoomed)", outfilename + "_zoom", zoom=True)

