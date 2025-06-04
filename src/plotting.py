# plotting.py

import geopandas as gpd
import folium

def plot_shapefile_with_labels(shapefile_path: str, map_location: list, map_zoom: int = 12, name_column: str = 'NAME'):
    # Load the shapefile using geopandas
    gdf = gpd.read_file(shapefile_path)
    
    # Reproject the shapefile to WGS84 if necessary
    if gdf.crs != "EPSG:4326":
        print(f"Reprojecting from {gdf.crs} to WGS84 (EPSG:4326)")
        gdf = gdf.to_crs(epsg=4326)
    
    # Create a folium map centered at map_location with the given zoom level
    folium_map = folium.Map(location=map_location, zoom_start=map_zoom)
    
    # Add each shape in the GeoDataFrame to the folium map
    for _, row in gdf.iterrows():
        # Get the geometry of the shape
        geom = row['geometry']
        
        # Get the value of the name column for this shape
        label = row[name_column] if name_column in row else "No Name"
        
        # Convert to GeoJson and add to the map with popup and tooltip
        folium.GeoJson(
            geom,
            tooltip=folium.Tooltip(label),  # Tooltip for mouse hover
            popup=folium.Popup(label)       # Popup when clicking
        ).add_to(folium_map)
    
    # Return the folium map object
    return folium_map

