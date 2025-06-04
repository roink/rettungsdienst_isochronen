import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter  
import contextily as ctx
import pyproj
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------------
# Define the suffixes and their corresponding titles
# -----------------------------------------------------------------------------
suffixes = [
    "base",
    "feiertag_false",
    "feiertag_true",
    "ferien_false",
    "ferien_true",
    "snowcover0.8_temp0.0",
    "uhrzeit_0",
    "uhrzeit_4",
    "uhrzeit_8",
    "uhrzeit_12",
    "uhrzeit_16",
    "uhrzeit_20",
    "wochentag_saturday",
    "wochentag_sunday",
    "wochentag_weekdayrandom",
]

title_map = {
    "base":                    "Erreichbarkeit",
    "feiertag_true":          "Erreichbarkeit Feiertag",
    "feiertag_false":         "Erreichbarkeit kein Feiertag",
    "ferien_true":            "Erreichbarkeit Ferien",
    "ferien_false":           "Erreichbarkeit keine Ferien",
    "snowcover0.8_temp0.0":   "Erreichbarkeit Schneedecke 0.8, Temp 0.0",
    "uhrzeit_0":              "Erreichbarkeit 0 Uhr",
    "uhrzeit_4":              "Erreichbarkeit 4 Uhr",
    "uhrzeit_8":              "Erreichbarkeit 8 Uhr",
    "uhrzeit_12":             "Erreichbarkeit 12 Uhr",
    "uhrzeit_16":             "Erreichbarkeit 16 Uhr",
    "uhrzeit_20":             "Erreichbarkeit 20 Uhr",
    "wochentag_saturday":     "Erreichbarkeit Samstags",
    "wochentag_sunday":       "Erreichbarkeit Sonntags",
    "wochentag_weekdayrandom":"Erreichbarkeit Wochentags",
}

# -----------------------------------------------------------------------------
# Function to plot the 0.9 probability contours for both 8min and 12min files
# -----------------------------------------------------------------------------
def plot_probability_contours(
    nc_8min,
    nc_12min,
    title_str,
    outpath,
    varname="prob",  
    sigma=None      
):
    """
    Plots the 0.9 contour from the 8-min and 12-min probability files on
    the same basemap, with red=8min and blue=12min. Removes x/y labels and
    adds a legend with '8 min' and '12 min'. Title is set to title_str.
    """
    # -------------------------------------------------------------------------
    # Open both datasets and extract the probability data
    # -------------------------------------------------------------------------
    ds8 = xr.open_dataset(nc_8min)
    ds12 = xr.open_dataset(nc_12min)
    
    prob8 = ds8[varname].fillna(0.0)
    prob12 = ds12[varname].fillna(0.0)

    # Coordinates
    lon = ds8["x"]
    lat = ds8["y"]
    
    # -------------------------------------------------------------------------
    # Create figure and axis
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create 2D meshgrid
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Transform from EPSG:4326 to EPSG:3857
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x2d, y2d = transformer.transform(lon2d, lat2d)

    # -------------------------------------------------------------------------
    # Optionally apply a Gaussian filter for smoothing
    # -------------------------------------------------------------------------
    data8 = prob8.values
    data12 = prob12.values
    if sigma is not None:
        data8 = gaussian_filter(data8, sigma=sigma)
        data12 = gaussian_filter(data12, sigma=sigma)

    # -------------------------------------------------------------------------
    # Draw contour lines at probability = 0.9
    # -------------------------------------------------------------------------
    cont8 = ax.contour(x2d, y2d, data8, levels=[0.9], colors="red", linewidths=2)
    cont12 = ax.contour(x2d, y2d, data12, levels=[0.9], colors="blue", linewidths=2)

    # We do NOT label the contours themselves (no ax.clabel)

    # -------------------------------------------------------------------------
    # Add the basemap underneath
    # -------------------------------------------------------------------------
    provider = ctx.providers.Stadia.StamenTonerLite(api_key="b0cd7fb3-b785-4366-8522-4269cc2721e4")
    provider["url"] = provider["url"] + "?api_key={api_key}"
    ctx.add_basemap(ax, source=provider, crs="EPSG:3857")

    # -------------------------------------------------------------------------
    # Legend, Title, and removing axis ticks/labels
    # -------------------------------------------------------------------------
    legend_elements = [
        Line2D([0], [0], color="red",  lw=2, label="8 min"),
        Line2D([0], [0], color="blue", lw=2, label="12 min")
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    ax.set_title(title_str)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # -------------------------------------------------------------------------
    # Set the axis extent to the entire domain
    # -------------------------------------------------------------------------
    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # -------------------------------------------------------------------------
    # Save the figure
    # -------------------------------------------------------------------------
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main script: loop over suffixes, produce 15 plots
# -----------------------------------------------------------------------------
for s in suffixes:
    file_8min = f"../data/processed/prob8min_{s}.nc"
    file_12min = f"../data/processed/prob12min_{s}.nc"

    # Define output filename
    outpath = f"../reports/figures/prob_contours_{s}.png"

    # Lookup the title; default to suffix if not found
    plot_title = title_map.get(s, s)

    # Generate the plot
    plot_probability_contours(
        nc_8min=file_8min,
        nc_12min=file_12min,
        title_str=plot_title,
        outpath=outpath,
        varname="mean_prediction",  # adjust if needed
        sigma=2       
    )
