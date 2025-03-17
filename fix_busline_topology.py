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
    half_drift = drift / 2
    # Iterate through each array in the list
    for line in lines_geom:
        print(f"\r{len(new_geom)}/{lines_geom.shape[0]}", end='')
        modified = False
        arr = line.coords._coords
        target_elements = np.where(np.isin(arr, visited))[0]
        if target_elements.shape[0]:
            modified = True
            angle = np.random.uniform(0, 2 * np.pi)
            drift_value = half_drift * (np.random.random() * 0.1 + 1)
            drift_vec = np.array((drift_value * np.cos(angle), drift_value * np.sin(angle)))
            arr[target_elements] += drift_vec
        
        # Find unique rows, their indices, and counts per unique coordinate.
        _, inverse, counts = np.unique(arr, axis=0, return_inverse=True, return_counts=True)
        # Create a boolean mask: True for coordinates that occur more than once.
        duplicate_mask = counts[inverse] > 1
        # For each duplicate instance, generate a random angle between 0 and 2*pi.
        num_duplicates = np.sum(duplicate_mask)
        if num_duplicates:
            modified = True
            random_angles = np.random.choice(np.linspace(0, 2*np.pi, num_duplicates * 10), size=num_duplicates, replace=False)
            # Compute the offset vector (of magnitude drift/2) for each duplicate.
            drift_vec = half_drift * np.column_stack((np.cos(random_angles), np.sin(random_angles)))
            # Add the offset only to rows that are duplicates.
            arr[duplicate_mask] += drift_vec
        
        if modified:
            new_geom.append(linestrings(arr))
        else:
            new_geom.append(line)
        visited = np.vstack((visited, arr))
    print()
    return new_geom

# Read the shapefile containing bus lines
bus_lines = gpd.read_file('bus_lines_filtered.shp')

# Apply the adjustment function to each linestring in the geometry column
bus_lines['geometry'] = make_unique_by_drift(bus_lines.geometry, 0.1)

# Save the modified data to a new shapefile
bus_lines.to_file('bus_lines_fixed.shp')

# Optional: Print confirmation
print("Topology fixed and saved to 'bus_lines_fixed.shp'")