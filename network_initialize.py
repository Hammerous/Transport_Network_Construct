import time, os, json
import concurrent.futures
import tools.numpy_cal as npcal
import tools.shp_process as shp
import tools.MeanShift_accelerate as ms_acc
import tools.networkX_manipulate as ntx

# 1. Points csv, index field name, coordinate field names, coordinate system’s EPSG serial
# (accept multiple inputs) 
# (Although it is rationally acceptable without a point input, detective points are requested for this mission)
pt_csv_param = [(r'grid_centroids.csv', 'Id', ('X', 'Y'), 'epsg:32651'),
                (r'bus_stops.csv', 'Bus_Id', ('X', 'Y'), 'epsg:4326'),
                (r'subway_stops.csv', 'Sub_Id', ('X', 'Y'), 'epsg:32651')]
                
# 2. Lines shapefile, preserved fields (list format)
# (only accept one file)
line_shp_path = r'walk_lines.shp'
preserve_fields = ['W_Id', 'Direction']

# 3. Targeted EPSG serial
# (Must be a projection coordinate system)
target_crs = "epsg:32651"

# 4. Buffer range to intersect line networks
buffer_range = 1500       # meter

# 5. Parameters for multi-thread acceleration
acc_param = {
    'blocks max thread': 16,
    'single max thread': 128,
    'bandwidth': 2000,         # meter, optional, 把多少米范围内的激发源对象视为同一block以参与后续运算
    'min_block': 2             # optional,一个block下至少有多少个点（至少为1） 
}

def initialize_points(pt_csv_param: tuple, target_crs: str, bandwidth=None, min_block=None):
    # Initialize the PointsLoader with CSV parameters and chunk size
    loader = shp.PointLoader(pt_csv_param, target_crs, chunksize=1e4)
    loader.load_all_csvs()
    print("Point Files Loaded !!!")
    coords_clusters, orphans_clusters = ms_acc.point_cluster(bandwidth, min_block,\
                                                             loader.gdf.geometry.get_coordinates().to_numpy())
    # Return a dictionary with the results
    if coords_clusters:
        print("Points Clustered !!!")
    return {
        'points': loader.gdf,
        'coords_clusters': coords_clusters,
        'orphans_clusters': orphans_clusters
    }

def initialize_lines(line_shp_path: str, preserve_fields: list, target_crs: str):
    lines = shp.LineLoader(line_shp_path, preserve_fields, target_crs)
    lines.load_lines()
    print("Line File Loaded !!!")
    return {
        'lines': lines.gdf,
        'nodes': lines.node_dict
    }

def find_projections(pt_idxs, buffer_range):
    global lines, points, processed_points
    pt_geom = points.iloc[pt_idxs].geometry
    status, nearby_lines_index, nearby_lines_arr = shp.nearby_lines(lines.geometry, pt_geom, buffer_range)
    processed_points += pt_idxs.shape[0]
    print(f'\rPoint in process: {processed_points}/{points.shape[0]}               ', end='')
    if status:
        ### all calculation is performed under UTM projection coordinate system
        pt_arr = pt_geom.geometry.get_coordinates().to_numpy()
        min_indices, attr_arr = npcal.find_nearest_intersect(pt_arr, nearby_lines_arr)
        return npcal.np.column_stack((pt_idxs, nearby_lines_index[min_indices])), attr_arr
    else:
        return pt_idxs, None

if __name__ == "__main__":
    dirpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dirpath)
    start_time = time.time()

    print("Initializing Files ...")
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
            print(f"{points.shape[0]} Points, {lines.shape[0]} Lines and {len(nodes)} Nodes Loaded")
            del future_points, future_lines
        except Exception as e:
            print(f"An error occurred: {e}")
            exit(1)

    print("Computing Points to Nearest Lines ...")
    # Launch threads for projection computations.
    results_list = []
    processed_points = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=acc_param['blocks max thread']) as executor:
        futures = [executor.submit(find_projections, pt_idxs, buffer_range)
                   for pt_idxs in coords_clusters.values()]
        for future in concurrent.futures.as_completed(futures):
            try:
                results_list.append(future.result())
            except Exception as exc:
                print(f"Exception in thread: {exc}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=acc_param["single max thread"]) as executor:
        futures = [executor.submit(find_projections, pt_idxs, buffer_range)
                   for pt_idxs in orphans_clusters.values()]
        for future in concurrent.futures.as_completed(futures):
            try:
                results_list.append(future.result())
            except Exception as exc:
                print(f"Exception in orphan processing thread: {exc}")

    print("\nCollecting Results ...")
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
    prj_gdf = shp.arr2gdf(attrs_lst, attrs_lst[:,2], attrs_lst[:,3], ['prj_length', 'dist', 'X', 'Y'], target_crs)
    prj_gdf['pt_id'] = points.iloc[idxs_lst[:,0]].index
    prj_gdf['line_id'] = lines.iloc[idxs_lst[:,1]].index
    del idxs_lst, attrs_lst
    print("Results Collected !!!")

    print("Building Topology ...")
    lines = shp.create_edges(prj_gdf, points, lines)
    print("Converting to Edgelist ...")
    edges = tuple(map(tuple, lines[['scr_encode', 'end_encode', 'length']].to_numpy()))
    print("Creating Graph ...")
    ntx.create_network_graph(edges, False, "test")

    print("Saving Files ...")
    # Only write to file if errors occurred
    if error_ids:
        error_log_file = "error_ids.txt"
        with open(error_log_file, "w") as file:
            file.write("\n".join(error_ids))
        print(f"Error IDs saved to {error_log_file}")
    
    # Save dictionary as a JSON file
    with open("line_nodes.json", "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=4)
    points.to_file("points.shp", driver="ESRI Shapefile", encoding='utf-8')
    prj_gdf.to_file("prj_pts.shp", driver="ESRI Shapefile", encoding='utf-8')
    lines.to_file("lines.shp", driver="ESRI Shapefile", encoding='utf-8')
    print('Files Saved !!!')

    print(f"Program ends in {time.time() - start_time:.2f} seconds")