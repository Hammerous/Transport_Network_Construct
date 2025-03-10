import time, os, json
import concurrent.futures
import tools.numpy_cal as npcal
import tools.coord_sys as crd
import tools.shp_process as shp
import tools.networkX_manipulate as ntx
import tools.MeanShift_accelerate as ms_acc

# 1. Points csv, index field name, coordinate field names, coordinate system’s EPSG serial
# (accept multiple inputs) 
# (Although it is rationally acceptable without a point input, detective points are requested for this mission)
pt_csv_param = [(r'grid_centroids.csv', 'Id', ('X', 'Y'), 'epsg:32651'),
                (r'bus_stops.csv', 'Bus_Id', ('X', 'Y'), 'epsg:4326'),
                (r'subway_stops.csv', 'Sub_Id', ('X', 'Y'), 'epsg:32651')]

# pt_csv_param = [(r'subway_stops.csv', 'Sub_Id', ('X', 'Y'), 'epsg:32651')]

# 2. Lines shapefile, index field name, direction field name (0 for no / 1 for has)
# (only accept one file)
line_shp_path = r'walk_lines.shp'
line_id_field = 'W_Id'
line_dir_field = 'Direction'

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

def initialize_points(pt_csv_param: dict, target_crs: str, bandwidth=None, min_block=None):
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

def initialize_lines(line_shp_path: str, idx_fields: list, target_crs: str):
    lines = shp.LineLoader(line_shp_path, idx_fields, target_crs)
    lines.load_lines()
    print("Line File Loaded, Saving ...")
    # Save dictionary as a JSON file
    with open("line_nodes.json", "w", encoding="utf-8") as f:
        json.dump(lines.node_dict, f, ensure_ascii=False, indent=4)
    # Return a dictionary with the results
    print("Line File Saved !!!")
    return {
        'lines': lines.gdf,
        'count': len(lines.node_dict)
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
        future_lines = executor.submit(initialize_lines, line_shp_path, [line_id_field, line_dir_field], target_crs)
        try:
            # Merge results from both functions
            points = future_points.result().get('points')
            coords_clusters = future_points.result().get('coords_clusters')
            orphans_clusters = future_points.result().get('orphans_clusters')
            lines = future_lines.result().get('lines')
            print(f"{points.shape[0]} Points, {lines.shape[0]} Lines and {future_lines.result().get('count')} Nodes Loaded")
            del future_points, future_lines
        except Exception as e:
            print(f"An error occurred: {e}")
            exit(1)

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

    idxs_list = []
    attrs_list = []
    for idx_arr, attr_arr in results_list:
        if attr_arr is None:
            print(f"Point: {points.iloc[idx_arr].index} not connected to network")
        else:
            idxs_list.append(idx_arr)
            attrs_list.append(attr_arr)
    idxs_list = npcal.np.concatenate(idxs_list)
    attrs_list = npcal.np.concatenate(attrs_list)
    print("\nSaving Files ...")
    prj_gdf = shp.gpd.GeoDataFrame(attrs_list, geometry=shp.gpd.points_from_xy(attrs_list[:,2],attrs_list[:,3]),\
                                   columns=['prj_length', 'distance', 'X', 'Y'], crs=target_crs)
    prj_gdf['pt_id'] = points.iloc[idxs_list[:,0]].index
    prj_gdf['line_id'] = lines.iloc[idxs_list[:,1]][line_id_field].values
    prj_gdf.to_file("prj_pts.shp", driver="ESRI Shapefile", encoding='utf-8')
    lines.to_file("lines.shp", driver="ESRI Shapefile", encoding='utf-8')
    exit()

    print("\nJoining child threads ...")
    for thread in threads:
        thread.join()
    if error_flag.is_set():
        print("An error occurred in one of the threads. Exiting without saving.")
    else:
        # Delete the results
        del points, pt_coordinates, coords_clusters, lines, array_lines, all_line_vecs
        print("Building Road Topology ...")
        idx_array = []
        float_array = []
        while not result.empty():
            ### nearby_lines_index, pt_idx, proj_length, distances
            idx_arr, float_arr = result.get()
            idx_array.append(idx_arr)
            float_array.append(float_arr)
        # 将列表转换为DataFrame
        idx_array = np.concatenate(idx_array)
        float_array = np.concatenate(float_array)
        line_idx_arr, proj_idx_arr = np.unique(idx_array[:, 0], return_inverse=True)

        # 将 points 和最近的垂足添加到图中，并连接它们
        edges_to_add = []
        edges_to_remove = []
        for idx, line_idx in enumerate(line_idx_arr):
            start, end = encoded_line_vectors[line_idx]
            nodes_index_arr = np.where(proj_idx_arr == idx)[0]

            ### 处理最近节点在线段两端的情况
            # 这里加入的是点在线段左下端时，节点直接连接线段端点的情况
            mask_zero = float_array[nodes_index_arr, 0] < 0
            edges_to_add.extend(
                                (pt_encoded[idx_array[pt_idx, 1]], start, float_array[pt_idx, 1])
                                for pt_idx in nodes_index_arr[mask_zero]
                            )
            # 这里加入的是点在线段右上端时，节点直接连接线段端点的情况
            mask_one = float_array[nodes_index_arr, 0] > 1
            edges_to_add.extend(
                                (pt_encoded[idx_array[pt_idx, 1]], end, float_array[pt_idx, 1])
                                for pt_idx in nodes_index_arr[mask_one]
                            )
            
            ### 处理最近节点为到线段垂足的情况
            feet_nodes_idx = nodes_index_arr[~(mask_zero | mask_one)]
            # 按顺序连接节点
            if feet_nodes_idx.shape[0]:
                edge_length = all_line_length[line_idx]
                edges_to_remove.append(line_idx)
                previous_point = start
                previous_position = 0
                if feet_nodes_idx.shape[0] > 1:
                    feet_nodes_idx = feet_nodes_idx[np.argsort(float_array[feet_nodes_idx, 0])]    #空间重新排序
                # 按顺序连接节点
                for pt_idx in feet_nodes_idx:
                    foot = f"{idx_array[pt_idx, 0]}-{pt_encoded[idx_array[pt_idx, 1]]}"
                    edges_to_add.extend((
                        (previous_point, foot, edge_length * (float_array[pt_idx, 0] - previous_position)),           ### 线段上垂足依次连接
                        (foot, pt_encoded[idx_array[pt_idx, 1]], float_array[pt_idx, 1])                                         ### 节点连接到垂足上
                    ))
                    ### 后期如果只需要计算纯路网距离，把和节点有关的weight设成0或者任意已知数值就行，最后求出来的结果返回去相减两次就能得到纯路网距离
                    previous_point = foot
                    previous_position = float_array[pt_idx, 0]
                # 连接最后一个点到终点
                edges_to_add.append((previous_point, end, edge_length * (1 - previous_position)))

        G = nx.Graph()
        #将合并后的数组转换为带权重的边列表, 批量插入边和权重
        G.add_weighted_edges_from([(encoded_line_vectors[i, 0], encoded_line_vectors[i, 1], all_line_length[i])\
                                    for i in np.setdiff1d(np.arange(encoded_line_vectors.shape[0]), edges_to_remove)]\
                                    + edges_to_add)
        print('Saving Results ... ')
        file_name_src = os.path.splitext(os.path.basename(source_csv_path))[0]
        file_name_trg = os.path.splitext(os.path.basename(target_csv_path))[0]
        # 保存到文件
        nx.write_weighted_edgelist(G, f"{file_name_src}-{file_name_trg}-faster.edgelist")
        #nx.write_gml(G, f"{file_name_src}-{file_name_trg}.gml")
        #nx.write_graphml(G, f"{file_name_src}-{file_name_trg}.graphml")
        print('Results Saved !!!')
    print(f"Program ends in {time.time() - start_time:.2f} seconds")