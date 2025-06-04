#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import ttest_ind

# Define file paths
final_dataset_path = "../data/interim/selected-data.parquet"
wohnbezirke_path = "../data/raw/Wohnbezirke_Hagen.shp"

# Load the dataset and create a GeoDataFrame of points
points_df = pd.read_parquet(final_dataset_path)
points_df.rename(columns={'Eintreffzeit': 'Hilfsfrist'}, inplace=True)

points_gdf = gpd.GeoDataFrame(
    points_df,
    geometry=gpd.points_from_xy(points_df.EINSATZORT_lon, points_df.EINSATZORT_lat),
    crs="EPSG:4326"
)

# ------------------------------------------------
# Construct "Wohnbezirk" column via spatial join
# ------------------------------------------------
# Load the Wohnbezirk shapefile and reproject if needed
districts_gdf = gpd.read_file(wohnbezirke_path)
if districts_gdf.crs != "EPSG:4326":
    districts_gdf = districts_gdf.to_crs(epsg=4326)

# Spatial join: assign each point to its corresponding district
joined_gdf = gpd.sjoin(points_gdf, districts_gdf, how="inner", predicate="within")
# Rename the district name column to "Wohnbezirk"
joined_gdf = joined_gdf.rename(columns={"NAME": "Wohnbezirk"})

def aggregate_by_year(data):
    """
    For each year, calculate:
      - Total incident count.
      - Count and percentage of incidents with Hilfsfrist <= 8 minutes.
      - Count and percentage of incidents with Hilfsfrist <= 12 minutes.
      - Average and standard deviation of Hilfsfrist.
    """
    # Total count per year
    total_counts = data.groupby("Jahr").size().rename("total_count")
    
    # Count of incidents with Hilfsfrist <= 8 minutes per year
    counts_leq_8 = data[data["Hilfsfrist"] <= 8].groupby("Jahr").size().rename("count_leq_8")
    
    # Count of incidents with Hilfsfrist <= 12 minutes per year
    counts_leq_12 = data[data["Hilfsfrist"] <= 12].groupby("Jahr").size().rename("count_leq_12")
    
    # Average and standard deviation of Hilfsfrist per year
    avg_Hilfsfrist = data.groupby("Jahr")["Hilfsfrist"].mean().rename("average_Hilfsfrist")
    avg_Fahrzeit = data.groupby("Jahr")["Fahrzeit"].mean().rename("average_Fahrzeit")
    avg_Dispositionszeit = data.groupby("Jahr")["Dispositionszeit"].mean().rename("average_Dispositionszeit")
    avg_Ausrückzeit = data.groupby("Jahr")["Ausrückzeit"].mean().rename("average_Ausrückzeit")
    
    # Combine the aggregated metrics into a DataFrame
    agg_df = pd.concat([total_counts, counts_leq_8, counts_leq_12, avg_Hilfsfrist, avg_Fahrzeit,  avg_Dispositionszeit, avg_Ausrückzeit], axis=1).reset_index()
    agg_df["count_leq_8"] = agg_df["count_leq_8"].fillna(0)
    agg_df["count_leq_12"] = agg_df["count_leq_12"].fillna(0)
    
    # Calculate percentages
    agg_df["percentage_leq_8"] = (agg_df["count_leq_8"] / agg_df["total_count"]) * 100
    agg_df["percentage_leq_12"] = (agg_df["count_leq_12"] / agg_df["total_count"]) * 100
    
    return agg_df

yearly_stats = aggregate_by_year(points_df)
print("Aggregated Hilfsfrist by Jahr:")
print(yearly_stats)

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

# Define a helper function to add a regression line (no legend label)
def add_regression(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    plt.plot(x, intercept + slope * x, linestyle='--', color='gray', alpha=0.7)
    return slope, p_value

# Start plot
plt.figure(figsize=(6, 4))

x = yearly_stats["Jahr"]

# Plot actual data
y = yearly_stats["average_Hilfsfrist"]
plt.plot(x, y, marker='o', label="Hilfsfrist")
add_regression(x, y)
y = yearly_stats["average_Fahrzeit"]
plt.plot(x, y, marker='o', label="Anfahrtszeit")
add_regression(x, y)
y = yearly_stats["average_Dispositionszeit"]
plt.plot(x, y, marker='o', label="Dispositionszeit")
add_regression(x, y)
y = yearly_stats["average_Ausrückzeit"]
plt.plot(x, y, marker='o', label="Ausrückzeit")
add_regression(x, y)

# Add a single dummy line for the regression trend in the legend
plt.plot([], [], linestyle='--', color='gray', label='')
plt.ylim(bottom=0)
# Plot formatting
plt.xlabel("Jahr")
plt.ylabel("Zeit [min]")
plt.title("Durchschnittliche Zeiten pro Jahr")
plt.grid(True)
plt.legend(loc='center left',bbox_to_anchor=(0.65, 0.75))
plt.tight_layout()

# Save the plot as a PNG file
plot_output_path = "../reports/figures/1.13-Zeit-Anteile.png"
plt.savefig(plot_output_path, dpi=300)
plt.show()

import pandas as pd
from scipy.stats import linregress

# Store results
results = []

# Loop through each attribute and compute regression
x = yearly_stats["Jahr"]
for col in ["average_Hilfsfrist", "average_Fahrzeit", "average_Dispositionszeit", "average_Ausrückzeit"]:
    y = yearly_stats[col]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    results.append({
        "Attribut": col,
        "Steigung": slope,
        "p-Wert": p_value
    })

# Create DataFrame and format
regression_table = pd.DataFrame(results)
regression_table["Steigung"] = regression_table["Steigung"].round(2)
regression_table["p-Wert"] = regression_table["p-Wert"].apply(lambda p: f"{p:.3f}")

# Display
print(regression_table)

# ------------------------------------------------
# Part 2: Plot average Hilfsfrist over "Jahr" per Wohnbezirk
# ------------------------------------------------
# Calculate the average Hilfsfrist per district per year using joined data
district_yearly = joined_gdf.groupby(["Wohnbezirk", "Jahr"])["Hilfsfrist"].mean().reset_index()

# Calculate overall average Hilfsfrist per year (across all incidents)
overall_avg = points_df.groupby("Jahr")["Hilfsfrist"].mean().reset_index()

# Create the plot
plt.figure(figsize=(6, 4))

# Plot each Wohnbezirk as a soft grey line
for district in district_yearly["Wohnbezirk"].unique():
    district_data = district_yearly[district_yearly["Wohnbezirk"] == district]
    plt.plot(district_data["Jahr"], district_data["Hilfsfrist"],
             color="grey", alpha=0.5, linewidth=1)

# Overlay the overall average as a bold black line
plt.plot(overall_avg["Jahr"], overall_avg["Hilfsfrist"],
         color="black", linewidth=2, label="Ganz Hagen")

# Set plot labels and title
plt.xlabel("Jahr")
plt.ylabel("Durschnittliche Hilfsfrist [min]")
plt.title("Entwicklung der durschnittlichen Hilfrist in den Wohnbezirken")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as a PNG file
plot_output_path = "../reports/figures/1.13-Hilfsfrist_by_Jahr_and_Wohnbezirk.png"
plt.savefig(plot_output_path, dpi=300)
plt.show()

from scipy.stats import linregress

# List to store regression results for each district
district_regression = []

# Group by district and iterate over each group
for district, group in district_yearly.groupby("Wohnbezirk"):
    # Ensure the data is sorted by year (if not already)
    group = group.sort_values("Jahr")
    
    # Run linear regression: Jahr (year) as the independent variable and Hilfsfrist as the dependent variable.
    slope, intercept, r_value, p_value, std_err = linregress(group["Jahr"], group["Hilfsfrist"])
    
    district_regression.append({
        "Wohnbezirk": district,
        "Jährliche Veränderung (min)": slope,  # Slope: minutes change per year
        "p-Wert": p_value                     # p-value for the hypothesis test that slope==0
    })

# Create a DataFrame with the regression results
regression_df = pd.DataFrame(district_regression)

# Optionally round the slope and p-value for easier reading
regression_df["Jährliche Veränderung (min)"] = regression_df["Jährliche Veränderung (min)"].round(2)
regression_df["p-Wert"] = regression_df["p-Wert"].round(3)

# Print the resulting table
print("Regression results for each Wohnbezirk:")
print(regression_df)

# Significance level threshold.
alpha = 0.05

# Split the DataFrame into three tables:
sig_increase_df = regression_df[
    (regression_df["Jährliche Veränderung (min)"] > 0) & (regression_df["p-Wert"] < alpha)
]
sig_decrease_df = regression_df[
    (regression_df["Jährliche Veränderung (min)"] < 0) & (regression_df["p-Wert"] < alpha)
]
no_effect_df = regression_df[regression_df["p-Wert"] >= alpha]

# Optionally round the results for easier reading.
sig_increase_df["Jährliche Veränderung (min)"] = sig_increase_df["Jährliche Veränderung (min)"].round(2)
sig_increase_df["p-Wert"] = sig_increase_df["p-Wert"].round(3)

sig_decrease_df["Jährliche Veränderung (min)"] = sig_decrease_df["Jährliche Veränderung (min)"].round(2)
sig_decrease_df["p-Wert"] = sig_decrease_df["p-Wert"].round(3)

no_effect_df["Jährliche Veränderung (min)"] = no_effect_df["Jährliche Veränderung (min)"].round(2)
no_effect_df["p-Wert"] = no_effect_df["p-Wert"].round(3)

# Print the three tables.
print("Statistically Significant Increase in Hilfsfrist (p < 0.05 & positive slope):")
print(sig_increase_df)

print("\nStatistically Significant Decrease in Hilfsfrist (p < 0.05 & negative slope):")
print(sig_decrease_df)

print("\nNo Significant Effect on Hilfsfrist (p >= 0.05):")
print(no_effect_df)

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import contextily as ctx  # for basemap

def plot_choropleth(gdf, name_column, value_column, output_path, min_value, max_value, title="Map", cmap="YlOrRd"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Reproject GeoDataFrame to Web Mercator for basemap compatibility
    gdf = gdf.to_crs(epsg=3857)
    
    # Plot the choropleth using the specified value column
    gdf.plot(
        column=value_column,
        ax=ax,
        cmap=cmap,
        edgecolor='black',
        linewidth=0.5,
        vmin=min_value,
        vmax=max_value,
        alpha=0.6
    )
    
    # Add OpenStreetMap basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    ax.set_axis_off()
    ax.set_title(title, fontsize=15)
    
    # Create a colorbar with the same scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=min_value, vmax=max_value))
    sm._A = []  # dummy array for the scalar mappable
    cbar = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label("Jährliche Veränderung der Hilfsfrist [min/a]")
    
    plt.savefig(output_path,dpi=300,     pil_kwargs={  
        "quality":    20,
        "optimize":   True,
        "progressive": True
    }, bbox_inches='tight')
    plt.close(fig)

# --- Merge Regression Results with GeoData ---


# Example merge: align the column names if necessary (here we merge on the Wohnbezirk name)
merged_gdf = districts_gdf.merge(regression_df, left_on="NAME", right_on="Wohnbezirk", how="left")

# Convert the slope column to numeric if needed (in case it was previously converted to string)
merged_gdf["Jährliche Veränderung (min)"] = pd.to_numeric(merged_gdf["Jährliche Veränderung (min)"], errors='coerce')

# Determine the color scale for the slope values:
min_value = merged_gdf["Jährliche Veränderung (min)"].min()
max_value = merged_gdf["Jährliche Veränderung (min)"].max()


# --- Plotting the Choropleth ---
output_path = "../reports/figures/1.13-Slope_Hilfsfrist_Wohnbezirke.jpg"
plot_choropleth(
    gdf=merged_gdf,
    name_column="Wohnbezirk",      
    value_column="Jährliche Veränderung (min)",            # Column with the regression slope (annual change in Hilfsfrist)
    output_path=output_path,
    min_value=min_value,
    max_value=max_value,
    title="Jährliche Veränderung der Hilfsfrist",
    cmap="coolwarm"              
)

# -------------------------------
# Calculate percentage of incidents with Hilfsfrist <= 8 minutes per district and year
# -------------------------------
def calculate_percentage_by_district_year(data):
    """
    Calculate the percentage of incidents with Hilfsfrist <= 8 minutes 
    for each Wohnbezirk and Jahr.
    """
    group = data.groupby(["Wohnbezirk", "Jahr"])
    total = group.size().rename("total_count")
    count_leq8 = group.apply(lambda x: (x["Hilfsfrist"] <= 8).sum()).rename("count_leq8")
    percentage = (count_leq8 / total * 100).rename("percentage_leq8")
    return percentage.reset_index()

district_yearly_pct = calculate_percentage_by_district_year(joined_gdf)

# -------------------------------
# Calculate overall percentage by year for all data
# -------------------------------
def calculate_overall_percentage_by_year(data):
    """
    Calculate the overall percentage of incidents with Hilfsfrist <= 8 minutes 
    for each Jahr (ignoring district boundaries).
    """
    group = data.groupby("Jahr")
    total = group.size().rename("total_count")
    count_leq8 = group.apply(lambda x: (x["Hilfsfrist"] <= 8).sum()).rename("count_leq8")
    percentage = (count_leq8 / total * 100).rename("percentage_leq8")
    return percentage.reset_index()

overall_yearly_pct = calculate_overall_percentage_by_year(points_df)

# -------------------------------
# Create the plot
# -------------------------------
plt.figure(figsize=(6, 4))

# Plot each Wohnbezirk as a soft grey line
for district in district_yearly_pct["Wohnbezirk"].unique():
    district_data = district_yearly_pct[district_yearly_pct["Wohnbezirk"] == district]
    plt.plot(district_data["Jahr"], district_data["percentage_leq8"],
             color="grey", alpha=0.5, linewidth=1)

# Overlay the overall percentage as a bold black line
plt.plot(overall_yearly_pct["Jahr"], overall_yearly_pct["percentage_leq8"],
         color="black", linewidth=2, label="Ganz Hagen")

# Configure plot appearance
plt.xlabel("Jahr")
plt.title("Entwicklung des 8 Minuten Erreichungsgrads\nin den Wohnbezirken")
plt.ylabel("Erreichungsgrad [%]")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as a PNG file
plot_output_path = "../reports/figures/1.13-Percentage_Hilfsfrist_leq_8_by_Jahr_and_Wohnbezirk.png"
plt.savefig(plot_output_path, dpi=300)
plt.show()

# Use the overall percentage DataFrame (overall_yearly_pct)
x_overall = overall_yearly_pct["Jahr"]
y_overall = overall_yearly_pct["percentage_leq8"]

slope_overall, intercept_overall, r_value_overall, p_value_overall, std_err_overall = linregress(x_overall, y_overall)

print("Overall Regression (Ganz Hagen):")
print(f"  Slope: {slope_overall:.2f} percentage points/year")
print(f"  p-Wert: {p_value_overall:.3f}\n")

# -------------------------------
# District-wise Regression for Percentage of Incidents (Hilfsfrist <= 8 minutes)
# -------------------------------

district_regression_results = []

# Group the district-level percentage DataFrame by Wohnbezirk
for district, group in district_yearly_pct.groupby("Wohnbezirk"):
    # Sort by year to be on the safe side
    group_sorted = group.sort_values("Jahr")
    x = group_sorted["Jahr"]
    y = group_sorted["percentage_leq8"]
    
    # Fit linear regression for the district's percentage data.
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    district_regression_results.append({
        "Wohnbezirk": district,
        "Slope (points/year)": slope,
        "p-Wert": p_value
    })

# Convert the list into a DataFrame for presentation
district_regression_df = pd.DataFrame(district_regression_results)

# Round the slope and p-value for easier interpretation
district_regression_df["Slope (points/year)"] = district_regression_df["Slope (points/year)"].round(2)

print("District-wise Regression Results:")
print(district_regression_df)

# In[ ]:

# Significance level threshold.
alpha = 0.05

# Split the DataFrame into three tables:
sig_increase_df = district_regression_df[
    (district_regression_df["Slope (points/year)"] > 0) & (district_regression_df["p-Wert"] < alpha)
]
sig_decrease_df = district_regression_df[
    (district_regression_df["Slope (points/year)"] < 0) & (district_regression_df["p-Wert"] < alpha)
]
no_effect_df = district_regression_df[district_regression_df["p-Wert"] >= alpha]

# Optionally round the results for easier reading.
sig_increase_df["Slope (points/year)"] = sig_increase_df["Slope (points/year)"].round(2)
sig_increase_df["p-Wert"] = sig_increase_df["p-Wert"].round(3)

sig_decrease_df["Slope (points/year)"] = sig_decrease_df["Slope (points/year)"].round(2)
sig_decrease_df["p-Wert"] = sig_decrease_df["p-Wert"].round(3)

no_effect_df["Slope (points/year)"] = no_effect_df["Slope (points/year)"].round(2)
no_effect_df["p-Wert"] = no_effect_df["p-Wert"].round(3)

# Print the three tables.
print("Statistically Significant Increase in 8min Erreichungsgrad (p < 0.05 & positive slope):")
print(sig_increase_df)

print("\nStatistically Significant Decrease in 8min Erreichungsgrad (p < 0.05 & negative slope):")
print(sig_decrease_df)

print("\nNo Significant Effect on 8min Erreichungsgrad (p >= 0.05):")
print(no_effect_df)

# -------------------------------
# Calculate percentage of incidents with Hilfsfrist <= 12 minutes per district and year
# -------------------------------
def calculate_percentage_by_district_year(data):
    """
    Calculate the percentage of incidents with Hilfsfrist <= 12 minutes 
    for each Wohnbezirk and Jahr.
    """
    group = data.groupby(["Wohnbezirk", "Jahr"])
    total = group.size().rename("total_count")
    count_leq12 = group.apply(lambda x: (x["Hilfsfrist"] <= 12).sum()).rename("count_leq12")
    percentage = (count_leq12 / total * 100).rename("percentage_leq12")
    return percentage.reset_index()

district_yearly_pct = calculate_percentage_by_district_year(joined_gdf)

# -------------------------------
# Calculate overall percentage by year for all data
# -------------------------------
def calculate_overall_percentage_by_year(data):
    """
    Calculate the overall percentage of incidents with Hilfsfrist <= 12 minutes 
    for each Jahr (ignoring district boundaries).
    """
    group = data.groupby("Jahr")
    total = group.size().rename("total_count")
    count_leq12 = group.apply(lambda x: (x["Hilfsfrist"] <= 12).sum()).rename("count_leq12")
    percentage = (count_leq12 / total * 100).rename("percentage_leq12")
    return percentage.reset_index()

overall_yearly_pct = calculate_overall_percentage_by_year(points_df)

# -------------------------------
# Create the plot
# -------------------------------
plt.figure(figsize=(6, 4))

# Plot each Wohnbezirk as a soft grey line
for district in district_yearly_pct["Wohnbezirk"].unique():
    district_data = district_yearly_pct[district_yearly_pct["Wohnbezirk"] == district]
    plt.plot(district_data["Jahr"], district_data["percentage_leq12"],
             color="grey", alpha=0.5, linewidth=1)

# Overlay the overall percentage as a bold black line
plt.plot(overall_yearly_pct["Jahr"], overall_yearly_pct["percentage_leq12"],
         color="black", linewidth=2, label="Ganz Hagen")

# Configure plot appearance
plt.xlabel("Jahr")
plt.ylabel("Erreichungsgrad [%]")
plt.title("Entwicklung des 12 Minuten Erreichungsgrads\nin den Wohnbezirken")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as a PNG file
plot_output_path = "../reports/figures/1.13-Percentage_Hilfsfrist_leq_12_by_Jahr_and_Wohnbezirk.png"
plt.savefig(plot_output_path, dpi=300)
plt.show()

# Use the overall percentage DataFrame (overall_yearly_pct)
x_overall = overall_yearly_pct["Jahr"]
y_overall = overall_yearly_pct["percentage_leq12"]

slope_overall, intercept_overall, r_value_overall, p_value_overall, std_err_overall = linregress(x_overall, y_overall)

print("Overall Regression (Ganz Hagen):")
print(f"  Slope: {slope_overall:.2f} percentage points/year")
print(f"  p-Wert: {p_value_overall:.3f}\n")

# -------------------------------
# District-wise Regression for Percentage of Incidents (Hilfsfrist <= 8 minutes)
# -------------------------------

district_regression_results = []

# Group the district-level percentage DataFrame by Wohnbezirk
for district, group in district_yearly_pct.groupby("Wohnbezirk"):
    # Sort by year to be on the safe side
    group_sorted = group.sort_values("Jahr")
    x = group_sorted["Jahr"]
    y = group_sorted["percentage_leq12"]
    
    # Fit linear regression for the district's percentage data.
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    district_regression_results.append({
        "Wohnbezirk": district,
        "Slope (points/year)": slope,
        "p-Wert": p_value
    })

# Convert the list into a DataFrame for presentation
district_regression_df = pd.DataFrame(district_regression_results)

# Round the slope and p-value for easier interpretation
district_regression_df["Slope (points/year)"] = district_regression_df["Slope (points/year)"].round(2)

print("District-wise Regression Results:")
print(district_regression_df)

# Significance level threshold.
alpha = 0.05

# Split the DataFrame into three tables:
sig_increase_df = district_regression_df[
    (district_regression_df["Slope (points/year)"] > 0) & (district_regression_df["p-Wert"] < alpha)
]
sig_decrease_df = district_regression_df[
    (district_regression_df["Slope (points/year)"] < 0) & (district_regression_df["p-Wert"] < alpha)
]
no_effect_df = district_regression_df[district_regression_df["p-Wert"] >= alpha]

# Optionally round the results for easier reading.
sig_increase_df["Slope (points/year)"] = sig_increase_df["Slope (points/year)"].round(2)
sig_increase_df["p-Wert"] = sig_increase_df["p-Wert"].round(3)

sig_decrease_df["Slope (points/year)"] = sig_decrease_df["Slope (points/year)"].round(2)
sig_decrease_df["p-Wert"] = sig_decrease_df["p-Wert"].round(3)

no_effect_df["Slope (points/year)"] = no_effect_df["Slope (points/year)"].round(2)
no_effect_df["p-Wert"] = no_effect_df["p-Wert"].round(3)

# Print the three tables.
print("Statistically Significant Increase in 12min Erreichungsgrad (p < 0.05 & positive slope):")
print(sig_increase_df)

print("\nStatistically Significant Decrease in 12min Erreichungsgrad (p < 0.05 & negative slope):")
print(sig_decrease_df)

print("\nNo Significant Effect on 12min Erreichungsgrad (p >= 0.05):")
print(no_effect_df)

# In[ ]:

