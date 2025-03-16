import time, os
import concurrent.futures
import tools.numpy_cal as npcal
import tools.shp_process as shp
import tools.file_manage as fm

# 1. Points csv, index field name, coordinate field names, coordinate system’s EPSG serial
# (accept multiple inputs) 
# (Although it is rationally acceptable without a point input, detective points are requested for this mission)
pt_csv_param = [(r'grid_centroids.csv', 'Id', ('X', 'Y'), 'epsg:32651'),
                (r'bus_stops.csv', 'Bus_Id', ('X', 'Y'), 'epsg:4326'),
                (r'subway_stops.csv', 'Sub_Id', ('X', 'Y'), 'epsg:32651')]
                
# 2. Lines shapefile, preserved fields (list format), direction_field (optional)
# (only accept one file)
line_shp_path = r'walk_lines.shp'
network_prefix = 'W'
#preserve_fields = ['W_Id', 'Direction']
preserve_fields = ['Direction']
direction_field = 'Direction'

# 3. Targeted EPSG serial (Must be a projection coordinate system)
target_crs = "epsg:32651"

# 4. Buffer range to intersect line networks
buffer_range = 1500       # meter

# 5. Merge tolerance to set the projected points at line's endpoints
# larger tolerance would create more concise topology network
merge_tol = 1e-2          # meter

def initialize_points(pt_csv_param: tuple, target_crs: str) -> dict:
    """
    Initializes the PointsLoader with CSV parameters, clusters the points based on provided bandwidth and min_block,
    and returns a dictionary with the results.

    Parameters:
    - pt_csv_param: A tuple containing the path to the CSV file and an additional CSV parameter (e.g., separator).
    - target_crs: The target coordinate reference system (CRS) to which the points will be transformed.
    - bandwidth: (Optional) The bandwidth parameter for clustering.
    - min_block: (Optional) The minimum block size for clustering.

    Returns:
    - A dictionary with the following keys:
      - 'points': The GeoDataFrame containing the loaded points.
      - 'coords_clusters': Clusters of coordinates based on the given bandwidth and min_block.
      - 'orphans_clusters': Orphan clusters that couldn't be assigned to any main clusters.
    """
    # Initialize the PointsLoader with CSV parameters and chunk size
    loader = shp.PointLoader(pt_csv_param, target_crs, chunksize=1e4)
    loader.load_all_csvs()
    print("Point Files Loaded !!!")    
    return {
        'points': loader.gdf
    }

def initialize_lines(line_shp_path: str, preserve_fields: list, target_crs: str) -> dict:
    """
    Initializes and loads line geometries from a shapefile.

    Args:
        line_shp_path (str): Path to the line shapefile.
        preserve_fields (List[str]): List of attribute fields to retain.
        target_crs (str): Target coordinate reference system for transformation.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'lines' (gpd.GeoDataFrame): Loaded line geometries.
            - 'nodes' (dict): Dictionary of nodes extracted from the line data.

    Notes:
        - Uses `shp.LineLoader` to load the shapefile and extract topology.
        - Outputs a message when loading is complete.
    """
    lines = shp.LineLoader(line_shp_path, preserve_fields, target_crs, network_prefix)
    lines.load_lines()

    print("Line File Loaded !!!")

    return {
        'lines': lines.gdf,         # GeoDataFrame containing line geometries
        'nodes': lines.nodes        # GeoDataFrame of extracted nodes
    }

def timestamp():
    """Returns the current time formatted as a timestamp."""
    return time.strftime("[%Y-%m-%d %H:%M:%S]")

if __name__ == "__main__":
    dirpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dirpath)
    start_time = time.time()
    
    print(f"{timestamp()} Initializing Files ...")
    # Use ThreadPoolExecutor to manage threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_points = executor.submit(initialize_points, pt_csv_param, target_crs)
        future_lines = executor.submit(initialize_lines, line_shp_path, preserve_fields, target_crs)
        try:
            # Merge results from both functions
            points = future_points.result().get('points')
            lines = future_lines.result().get('lines')
            nodes = future_lines.result().get('nodes')
            print(f"{timestamp()} {points.shape[0]} Points, {lines.shape[0]} Lines and {nodes.shape[0]} Nodes Loaded")
            del future_points, future_lines
        except Exception as e:
            print(f"{timestamp()} An error occurred: {e}")
            exit(-1)

    print(f"{timestamp()} Computing Points to Nearest Lines ...")
    pt_idx, line_idx = shp.nearby_lines(lines.geometry, points.geometry, buffer_range)
    prj_attrs, prj_gdf = npcal.compute_proj_length(points.geometry.iloc[pt_idx].get_coordinates().to_numpy(),\
                                                     lines.geometry.iloc[line_idx].get_coordinates().to_numpy().reshape(-1,2,2))

    prj_gdf = shp.arr2gdf(prj_attrs, prj_gdf[:,0], prj_gdf[:,1], ['prj_length', 'to_src', 'to_end'], target_crs)
    prj_gdf['pt_id'] = points.iloc[pt_idx].index
    prj_gdf['line_id'] = lines.iloc[line_idx].index
    prj_gdf['this_node'] = network_prefix + prj_gdf['line_id'].astype(str) + '-' + prj_gdf['pt_id'].astype(str)
    error_ids = list(points.index.drop(points.index[pt_idx]))
    del pt_idx, line_idx, prj_attrs

    print(f"{timestamp()} Building Topology into Shapefile ...")
    lines = shp.create_edges(prj_gdf.copy(), points.copy(), lines.copy(), merge_tol, direction_field)

    print(f"{timestamp()} Creating Edgelist...")
    edges = shp.create_edgelist(lines, direction_field)

    print(f"{timestamp()} Saving Files ...")
    with open(fm.add_prefix(line_shp_path, "topo_").split(".")[0]+".edgelist", 'w') as f:
        f.write(''.join(f"{scr} {end} {length}\n" for scr, end, length in edges))
    
    # Only write to file if dropping occurred
    if error_ids:
        error_log_file = "error_ids.txt"
        with open(error_log_file, "w") as file:
            file.write("\n".join(error_ids))
        print(f"{timestamp()} Error IDs saved to {error_log_file}")

    points.to_file(fm.add_affix(line_shp_path, "_pts"), driver="ESRI Shapefile", encoding='utf-8')
    nodes.to_file(fm.add_affix(line_shp_path, "_nodes"), driver="ESRI Shapefile", encoding='utf-8')
    prj_gdf.to_file(fm.add_affix(line_shp_path, "_prjs"), driver="ESRI Shapefile", encoding='utf-8')
    lines.to_file(fm.add_prefix(line_shp_path, "topo_"), driver="ESRI Shapefile", encoding='utf-8')
    
    print(f"{timestamp()} Program ends in {time.time() - start_time:.2f} seconds, go check files in this script's folder")