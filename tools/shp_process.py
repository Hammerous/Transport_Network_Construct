import pandas as pd
import geopandas as gpd
from pyproj import CRS
import concurrent.futures
import MeanShift_accelerate as ms_acc

class PointLoader:
    def __init__(self, pt_csv_params, target_crs, chunksize=10000):
        """
        Initialize the loader with CSV parameters, target CRS, and chunk size.

        Parameters:
            pt_csv_params: Tuple of CSV parameters. Each element is a tuple:
                           (csv_file, index_col, (x_field, y_field), src_epsg)
            target_crs: pyproj.CRS object representing the target coordinate system.
            chunksize: Number of rows per chunk when reading CSV files.
        """
        self.pt_csv_params = pt_csv_params
        self.target_crs = CRS(target_crs)
        self.chunksize = chunksize

    def process_csv(self, file_param):
        """
        Process a single CSV file and return a GeoDataFrame.

        Steps:
         - Reads the CSV in chunks, loading only the necessary columns.
         - Creates a GeoDataFrame from each chunk with geometry built from x and y fields.
         - Sets the source CRS and transforms to target CRS if conversion is needed.
         - Renames the index column to a common name 'id'.
         - Keeps only the 'id' and 'geometry' columns.
        
        Parameters:
            file_param: Tuple (csv_file, index_col, (x_field, y_field), src_epsg)
        
        Returns:
            A concatenated GeoDataFrame for the entire CSV file.
        """
        csv_file, index_col, (x_field, y_field), src_epsg = file_param
        src_crs = CRS(src_epsg)
        chunks = []
        
        # Read CSV in chunks loading only the necessary columns.
        for chunk in pd.read_csv(csv_file, usecols=[index_col, x_field, y_field], chunksize=self.chunksize):
            # Create geometry using the x and y coordinate fields.
            gdf_chunk = gpd.GeoDataFrame(
                chunk,
                geometry=gpd.points_from_xy(chunk[x_field], chunk[y_field])
            )
            # Set the source coordinate system.
            gdf_chunk.set_crs(src_epsg, inplace=True)
            
            # If source and target CRS differ, perform coordinate transformation.
            if src_crs != self.target_crs:
                gdf_chunk = gdf_chunk.to_crs(self.target_crs)
            
            # Keep only the index column and geometry column.
            chunks.append(gdf_chunk[[index_col, 'geometry']])
        
        return pd.concat(chunks, ignore_index=True).rename(columns={index_col: "id"})

    def load_all_csvs(self):
        """
        Process all CSV files concurrently, verify no duplicate IDs exist across files,
        and concatenate them into a single GeoDataFrame.

        Raises:
            ValueError: If duplicate IDs are found in the combined GeoDataFrame.
        
        Returns:
            A single combined GeoDataFrame containing all processed points.
        """
        gdf_list = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Process each CSV concurrently.
            futures = [executor.submit(self.process_csv, param) for param in self.pt_csv_params]
            for future in concurrent.futures.as_completed(futures):
                gdf_list.append(future.result())
        
        # Concatenate all GeoDataFrames.
        combined_gdf = pd.concat(gdf_list, ignore_index=True)
        
        # Check for duplicate IDs.
        if combined_gdf["id"].duplicated().any():
            raise ValueError("Duplicate IDs found in the combined GeoDataFrame.")
        else:
            combined_gdf.set_index("id")
        
        return combined_gdf

# Define the first function for clustering
def initialize_points(pt_csv_param, result, target_crs,
                      bandwidth = None, num_in_block = None):
    # Initialize the PointsLoader with CSV parameters and chunk size
    loader = PointLoader(pt_csv_param, target_crs, chunksize=10000)
    gdf = loader.load_all_csvs()
    coords_clusters = ms_acc.point_cluster(bandwidth, num_in_block, gdf.geometry, True, False)  # -1 is for orphans
    orphans_clusters = {key: coords_clusters[-1][val] for key, val in\
                        point_cluster(bandwidth, 1, pt_coordinates[coords_clusters[-1]], True, True).items()}
    del coords_clusters[-1]
    
    result['points'] = points
    result['pt_coordinates'] = pt_coordinates
    result['coords_clusters'] = coords_clusters
    result['orphans_clusters'] = orphans_clusters

def loading_lines(lines, result):
    node_dict = {}
    count = 0
    def get_or_create_encoding(coord):
        nonlocal count
        if coord not in node_dict:
            node_dict[coord] = f'L{count}'
            count += 1
        return node_dict[coord]

    encoded_line_vectors = []
    array_lines = []
    for line in lines.geometry:
        for start, end in zip(line.coords[:-1], line.coords[1:]):
            encoded_line_vectors.append((get_or_create_encoding(start), get_or_create_encoding(end)))
            array_lines.append((start, end))

    lines = gpd.GeoDataFrame(geometry=[LineString(coords) for coords in array_lines], crs=lines.crs)
    # 将数据转换为numpy数组
    array_lines = np.array(array_lines, dtype=np.float64)
    encoded_line_vectors = np.array(encoded_line_vectors)

    # 根据start和end的坐标大小关系重新排列
    sorted_indices = np.lexsort((array_lines[:, :, 1], array_lines[:, :, 0]))
    array_lines = array_lines[np.arange(array_lines.shape[0])[:, None], sorted_indices]
    encoded_line_vectors = encoded_line_vectors[np.arange(encoded_line_vectors.shape[0])[:, None], sorted_indices]

    ### array_lines已经按照从左下到右上的顺序排列，encoded_line_vectors为对应顺序的线段端点
    all_line_vecs = array_lines[:, 1, :] - array_lines[:, 0, :]   # 计算线段的向量
    all_line_length = np.linalg.norm(all_line_vecs, axis=1)

    #返回数据
    result['lines'] = lines
    result['array_lines'] = array_lines
    result['encoded_line_vectors'] = encoded_line_vectors
    result['all_line_length'] = all_line_length
    result['all_line_vecs'] = all_line_vecs
    result['count'] = count

def get_nearby(lines, pt_arr, buffer_distance):
    global array_lines, all_line_vecs
    Status = False
    line_vectors = False
    line_vecs = False

    buffer = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pt_arr[:,0], pt_arr[:,1]),crs=lines.crs).buffer(buffer_distance).union_all()
    nearby_lines_index = lines[lines.intersects(buffer)].index.to_numpy()
    if nearby_lines_index.shape[0]:
        Status = True
        line_vectors = array_lines[nearby_lines_index]
        line_vecs = all_line_vecs[nearby_lines_index]
    return Status, line_vectors, line_vecs, nearby_lines_index