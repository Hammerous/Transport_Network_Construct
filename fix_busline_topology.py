import geopandas as gpd
import numpy as np
from shapely import linestrings
import numpy as np

def make_unique_by_drift(lines_geom, drift):
    """
    Process a list of NumPy arrays to ensure all elements are unique by adding a drift.
    
    Parameters:
    - arr_list: List of NumPy arrays, each with shape (n, 2), where n can vary.
    
    The function modifies the arrays in place. If an element (row) is a duplicate of one
    previously encountered, it adds (0.1, 0.1) to the element until it is unique.
    """
    # Initialize an empty set to track visited elements
    visited = np.empty(shape=(0,2),dtype=float)
    new_geom = []
    # Iterate through each array in the list
    for line in lines_geom:
        arr = line.coords._coords
        target_elements = np.where(np.isin(arr, visited))[0]
        if target_elements.shape[0]:
            angle = np.random.uniform(0, 2 * np.pi)
            drift_vec = np.array((drift * np.cos(angle), drift * np.sin(angle)))
            arr[target_elements] += drift_vec
            new_geom.append(linestrings(arr))
        else:
            new_geom.append(line)
        visited = np.vstack((visited, arr))
    return new_geom

# Read the shapefile containing bus lines
bus_lines = gpd.read_file('bus_lines_filtered.shp')

# Apply the adjustment function to each linestring in the geometry column
bus_lines['geometry'] = make_unique_by_drift(bus_lines.geometry, 0.1)

# Save the modified data to a new shapefile
bus_lines.to_file('bus_lines_fixed.shp')

# Optional: Print confirmation
print("Topology fixed and saved to 'bus_lines_fixed.shp'")