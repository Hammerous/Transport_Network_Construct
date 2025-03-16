import geopandas as gpd
from pyproj import CRS
import pandas as pd

bus_lines = gpd.read_file("bus_lines_fixed.shp")

bus_stops = pd.read_csv("bus_stops_filtered.csv")
bus_stops = gpd.GeoDataFrame(bus_stops, geometry=gpd.points_from_xy(bus_stops['X'], bus_stops['Y']), crs='epsg:4326')
bus_stops.to_crs(bus_lines.crs, inplace=True)

# Ensure both have the 'BusName' field
if 'BusName' not in bus_stops.columns or 'BusName' not in bus_lines.columns:
    raise ValueError("Missing 'BusName' field in one or both dataframes")

# Function to snap a point to the nearest point on a line
def snap_to_line(point, line):
    return line.interpolate(line.project(point))

# Create a copy of bus stops to store corrected locations
bus_stops_corrected = bus_stops.copy()

# Iterate through bus lines and adjust corresponding bus stops
for _, line in bus_lines.iterrows():
    print(f"\r{_}", end='')
    bus_name = line['Dir_Name']
    matching_stops = bus_stops_corrected[bus_stops_corrected['Dir_Name'] == bus_name]
    
    for idx, stop in matching_stops.iterrows():
        new_location = snap_to_line(stop.geometry, line.geometry)
        bus_stops_corrected.at[idx, 'geometry'] = new_location
print()

# Save to a new shapefile
bus_stops_corrected.to_file("bus_stops_fixed.shp")

print("Corrected bus stops saved successfully.")
