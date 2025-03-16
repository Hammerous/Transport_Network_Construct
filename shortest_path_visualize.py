import pandas as pd
import geopandas as gpd
from shapely import linestrings
import multiprocessing as mp
import ast, time

# Point files
pt_gdf = [
    ("walk_lines_pts.shp", "id"),
    ("walk_lines_nodes.shp", "Id"),
    ("walk_lines_prjs.shp", "this_node"),
    ("bus_lines_nodes.shp", "Id"),
    ("bus_lines_prjs.shp", "this_node"),
    ("subway_lines_nodes.shp", "Id"),
    ("subway_lines_prjs.shp", "this_node")
    ]

# OD path file
od_df_path = 'OD_file_in_Hybrid_Network.csv'  # Replace with the path to your CSV file

def num_worker_find(num_workers):
    return min(mp.cpu_count(), num_workers) if isinstance(num_workers, int) and num_workers > 0 else mp.cpu_count()

def read_shapefile(args):
    """Reads a shapefile and extracts only the geometry and specified ID column."""
    shp_path, id_col = args
    try:
        gdf = gpd.read_file(shp_path, columns=[id_col, "geometry"])
        return gdf.set_index(id_col)
    except Exception as e:
        print(f"Error reading {shp_path}: {e}")
        return None

def read_shapefiles_parallel(shp_list, num_workers=None):
    """
    Reads multiple shapefiles in parallel, extracting geometry and the given ID column.
    
    :param shp_list: List of tuples (shp_path, Id_col_str)
    :param num_workers: Number of processes (default: CPU count)
    :return: List of GeoDataFrames
    """
    num_workers = num_worker_find(num_workers)

    with mp.Pool(num_workers) as pool:
        results = pool.map(read_shapefile, shp_list)

    return pd.concat([gdf for gdf in results if gdf is not None], ignore_index=False)

def convert_path_to_linestring(point_ids, gdf):
    coordinates = gdf.loc[point_ids].get_coordinates().to_numpy()
    return linestrings(coordinates)

def create_geometry(df, gdf, num_workers=None):
    """ Apply function in parallel using multiprocessing. """
    num_workers = num_worker_find(num_workers)
    args = [(ast.literal_eval(path_str), gdf) for path_str in df['path'].to_list()]
    with mp.get_context().Pool(num_workers) as pool:
        path_linestrings = pool.starmap(convert_path_to_linestring, args)
    return path_linestrings

def timestamp():
    """Returns the current time formatted as a timestamp."""
    return time.strftime("[%Y-%m-%d %H:%M:%S]")

if __name__ == "__main__":
    print(f"{timestamp()} Reading Files ...")
    pt_gdf = read_shapefiles_parallel(pt_gdf, 8)
    od_df = pd.read_csv(od_df_path)

    print(f"{timestamp()} Converting Geometry ...")
    # Convert path to LineString in parallel
    od_df['geometry'] = create_geometry(od_df, pt_gdf, 8)
    # Convert to GeoDataFrame
    od_df = gpd.GeoDataFrame(od_df.drop(columns=['path']), geometry='geometry', crs=pt_gdf.crs)

    print(f"{timestamp()} Saving File ...")
    od_df.to_file(f'{od_df_path.split(".")[0]}.shp')
