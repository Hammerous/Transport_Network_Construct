import time, os, json, threading
import concurrent.futures
import tools.numpy_cal as npcal
import tools.shp_process as shp
import tools.MeanShift_accelerate as ms_acc
import tools.file_manage as fm

# 1. Points csv, index field name, coordinate field names, coordinate system’s EPSG serial
# (accept multiple inputs) 
# (Although it is rationally acceptable without a point input, detective points are requested for this mission)
pt_csv_param = [(r'grid_centroids.csv', 'Id', ('X', 'Y'), 'epsg:32651'),
                (r'bus_stops.csv', 'Bus_Id', ('X', 'Y'), 'epsg:4326'),
                (r'subway_stops.csv', 'Sub_Id', ('X', 'Y'), 'epsg:32651')]
                
# 2. Lines shapefile, preserved fields (list format)
# (only accept one file)
line_shp_path = r'walk_lines.shp'
#preserve_fields = ['W_Id', 'Direction']
preserve_fields = ['Direction']

# 3. Targeted EPSG serial (Must be a projection coordinate system)
target_crs = "epsg:32651"

# 4. Buffer range to intersect line networks
buffer_range = 1500       # meter

# 5. Parameters for multi-thread acceleration
acc_param = {
    'blocks max thread': 16,    # threads to process dense points blocks, recommended to reduce if running memory insufficienct
    'single max thread': 64,   # threads to process sparse points blocks or single points
    'bandwidth': 2000,          # optional, diameter in meters to create a points block
    'min_block': 2              # optional, minimun points to create a points block
}

def initialize_points(pt_csv_param: tuple, target_crs: str, bandwidth: float = None, min_block: int = None) -> dict:
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
    
    # Perform clustering on the loaded points
    coords_clusters, orphans_clusters = ms_acc.point_cluster(bandwidth, min_block,
                                                             loader.gdf.geometry.get_coordinates().to_numpy())
    
    # Return a dictionary with the results
    if coords_clusters:
        print("Points Clustered !!!")
    
    return {
        'points': loader.gdf,
        'coords_clusters': coords_clusters,
        'orphans_clusters': orphans_clusters
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
    lines = shp.LineLoader(line_shp_path, preserve_fields, target_crs)
    lines.load_lines()

    print("Line File Loaded !!!")

    return {
        'lines': lines.gdf,  # GeoDataFrame containing line geometries
        'nodes': lines.node_dict  # Dictionary of extracted nodes
    }

def find_projections(pt_idxs: npcal.np.ndarray, buffer_range: float) -> tuple[npcal.np.ndarray, npcal.np.ndarray]:
    """
    Finds the nearest projections of given points onto nearby lines within a specified buffer range.

    Args:
        pt_idxs (np.ndarray): Array of point indices to be processed.
        buffer_range (float): Search radius for finding nearby lines.

    Returns:
        tuple[npcal.np.ndarray, npcal.np.ndarray]
            - A 2D array in shape (n,2) where each row contains a point index and its nearest line index.
            - A 2D array in shape (n,3) of computed attributes (prj_length, X/Y coordinates), or None if no nearby lines are found.
    
    Global Variables:
        - lines: GeoDataFrame containing line geometries.
        - points: GeoDataFrame containing point geometries.
        - processed_points: Counter for tracking the number of processed points.
    
    Notes:
        - Assumes all calculations are performed in a UTM projection coordinate system.
        - Uses `shp.nearby_lines()` to identify nearby lines within the buffer range.
        - Utilizes `npcal.find_nearest_intersect()` for nearest point-line intersection calculations.
    """
    global points, lines, processed_points

    pt_geom = points.iloc[pt_idxs].geometry  # Extract geometries of the selected points
    processed_points += pt_idxs.shape[0]
    print(f'\rPoint in process: {processed_points}/{points.shape[0]}               ', end='')
    ### ⚠⚠⚠ Notice: lines are accessed by abundant threads, thus require memory protection. You could release this lock if memory permits
    with lock:
        nearby_lines_index, nearby_lines_arr = shp.nearby_lines(lines.geometry, pt_geom, buffer_range)

    # Determine the status based on whether any intersections were found.
    if nearby_lines_index.size > 0: 
        # Extract coordinate arrays from point geometries
        pt_arr = pt_geom.geometry.get_coordinates().to_numpy()
        min_indices, attr_arr = npcal.find_nearest_intersect(pt_arr, nearby_lines_arr)
        # Return point indices paired with their nearest line indices, along with computed attributes
        return npcal.np.column_stack((pt_idxs, nearby_lines_index[min_indices])), attr_arr
    else:
        return pt_idxs, None  # No nearby lines found

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
        future_points = executor.submit(initialize_points, pt_csv_param, target_crs,
                                          acc_param['bandwidth'], acc_param['min_block'])
        future_lines = executor.submit(initialize_lines, line_shp_path, preserve_fields, target_crs)
        try:
            # Merge results from both functions
            points = future_points.result().get('points')
            coords_clusters = future_points.result().get('coords_clusters')
            orphans_clusters = future_points.result().get('orphans_clusters')
            lines = future_lines.result().get('lines')
            nodes = future_lines.result().get('nodes')
            print(f"{timestamp()} {points.shape[0]} Points, {lines.shape[0]} Lines and {len(nodes)} Nodes Loaded")
            del future_points, future_lines
        except Exception as e:
            print(f"{timestamp()} An error occurred: {e}")
            exit(1)

    print(f"{timestamp()} Computing Points to Nearest Lines ...")
    # Launch threads for projection computations.
    results_list = []
    processed_points = 0
    ### ⚠⚠⚠ Notice: lines accessed by abundant threads, thus require memory protection. You could release this lock if memory permits
    lock = threading.Lock() ### this should be part of GIL
    with concurrent.futures.ThreadPoolExecutor(max_workers=acc_param['blocks max thread']) as executor:
        futures = [executor.submit(find_projections, pt_idxs, buffer_range)
                   for pt_idxs in coords_clusters.values()]
        for future in concurrent.futures.as_completed(futures):
            try:
                results_list.append(future.result())
            except Exception as exc:
                print(f"{timestamp()} Exception in thread: {exc}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=acc_param["single max thread"]) as executor:
        futures = [executor.submit(find_projections, pt_idxs, buffer_range)
                   for pt_idxs in orphans_clusters.values()]
        for future in concurrent.futures.as_completed(futures):
            try:
                results_list.append(future.result())
            except Exception as exc:
                print(f"{timestamp()} Exception in orphan processing thread: {exc}")
    print()
    idxs_lst = []
    attrs_lst = []
    error_ids = []
    for idx_arr, attr_arr in results_list:
        if attr_arr is None:
            error_ids.extend([i for i in points.iloc[idx_arr].index])
        else:
            idxs_lst.append(idx_arr)
            attrs_lst.append(attr_arr)
    
    idxs_lst = npcal.collect_arr(idxs_lst)
    attrs_lst = npcal.collect_arr(attrs_lst)
    prj_gdf = shp.arr2gdf(attrs_lst[:, 0], attrs_lst[:,1], attrs_lst[:,2], ['prj_length'], target_crs)
    prj_gdf['pt_id'] = points.iloc[idxs_lst[:,0]].index
    prj_gdf['line_id'] = lines.iloc[idxs_lst[:,1]].index
    del idxs_lst, attrs_lst

    print(f"{timestamp()} Building Topology into Shapefile ...")
    lines = shp.create_edges(prj_gdf, points, lines)

    print(f"{timestamp()} Saving Files ...")
    # Only write to file if errors occurred
    if error_ids:
        error_log_file = "error_ids.txt"
        with open(error_log_file, "w") as file:
            file.write("\n".join(error_ids))
        print(f"{timestamp()} Error IDs saved to {error_log_file}")
    
    # Save dictionary as a JSON file
    with open("line_nodes.json", "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=4)
    points.to_file(fm.add_affix(line_shp_path, "pts"), driver="ESRI Shapefile", encoding='utf-8')
    prj_gdf.to_file(fm.add_affix(line_shp_path, "prjs"), driver="ESRI Shapefile", encoding='utf-8')
    lines.to_file(fm.add_prefix(line_shp_path, "topo"), driver="ESRI Shapefile", encoding='utf-8')

    print(f"{timestamp()} Program ends in {time.time() - start_time:.2f} seconds, go check files in this script's folder")