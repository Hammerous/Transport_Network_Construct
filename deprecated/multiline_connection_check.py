import geopandas as gpd
from shapely.geometry import MultiLineString, LineString

# --- Parameters ---
# Path to the original shapefile
input_shp = r"D:\2025Spring\Accessibility\Public Transport\bus_lines.shp"
# Output shapefile for connected multi-line objects
output_connected = 'connected_lines.shp'
# Output shapefile for disconnected multi-line objects
output_disconnected = 'disconnected_lines.shp'

# --- Read input shapefile ---
gdf = gpd.read_file(input_shp)

# Lists to hold geometries based on connectivity
connected_rows = []
disconnected_rows = []

# --- Process each feature, preserving attributes ---
for idx, row in gdf.iterrows():
    geom = row.geometry
    # Handle both MultiLineString and LineString geometries.
    if geom.geom_type == 'MultiLineString':
        segments = list(geom.geoms)
    elif geom.geom_type == 'LineString':
        segments = [geom]
    else:
        # Skip geometries that are not line-based
        continue

    # Assume connected until a break is found.
    is_connected = True
    if len(segments) > 1:
        for i in range(len(segments) - 1):
            seg_current = segments[i]
            seg_next = segments[i + 1]
            # Check if the end of one segment matches the start of the next.
            if seg_current.coords[-1] != seg_next.coords[0]:
                is_connected = False
                break

    # Append the entire row (attributes + geometry) to the corresponding list.
    if is_connected:
        connected_rows.append(row)
    else:
        disconnected_rows.append(row)

# --- Create new GeoDataFrames preserving attributes ---
gdf_connected = gpd.GeoDataFrame(connected_rows, crs=gdf.crs)
gdf_disconnected = gpd.GeoDataFrame(disconnected_rows, crs=gdf.crs)

# --- Save outputs to new shapefiles ---
gdf_connected.to_file(output_connected)
gdf_disconnected.to_file(output_disconnected)

# --- Save the outputs to new shapefiles ---
gdf_connected.to_file(output_connected)
gdf_disconnected.to_file(output_disconnected)

print("Processing complete:")
print(f"  Connected geometries saved to: {output_connected}")
print(f"  Disconnected geometries saved to: {output_disconnected}")