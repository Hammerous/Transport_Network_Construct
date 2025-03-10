import pandas as pd
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString

class PointLoader:
    def __init__(self, pt_csv_params, target_crs, chunksize=10000, max_workers=8):
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
        self.max_workers = max_workers

    def process_csv(self, file_param):
        """
        Process a single CSV file and return a GeoDataFrame.

        Steps:
        - Reads the CSV in chunks, loading only the necessary columns.
        - Accumulates the chunks as plain DataFrames.
        - Concatenates all chunks and then creates the geometry column using x and y fields.
        - Sets the source CRS and transforms to the target CRS if needed.
        - Renames the index column to 'id' and keeps only 'id' and 'geometry' columns.
        
        Parameters:
            file_param: Tuple (csv_file, index_col, (x_field, y_field), src_epsg)
        
        Returns:
            A GeoDataFrame for the entire CSV file.
        """
        csv_file, index_col, (x_field, y_field), src_epsg = file_param
        src_crs = CRS(src_epsg)
        
        # Read CSV in chunks as plain DataFrames.
        df = [
            chunk for chunk in pd.read_csv(
                csv_file, usecols=[index_col, x_field, y_field], chunksize=self.chunksize
            )
        ]
        
        # Concatenate all chunks into one DataFrame.
        df = pd.concat(df, ignore_index=True)
        
        # Create the geometry column for the entire DataFrame at once.
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_field], df[y_field]))
        
        # Set the source CRS.
        gdf.set_crs(src_epsg, inplace=True)
        
        # Transform to the target CRS if needed.
        if src_crs != self.target_crs:
            gdf = gdf.to_crs(self.target_crs)
        
        # Rename the index column to "id".
        gdf.rename(columns={index_col: "id"}, inplace=True)
        return gdf[["id", 'geometry']]  # Keep only the index and geometry columns.

    def load_all_csvs(self):
        """
        Process all CSV files sequentially, verify no duplicate IDs exist across files,
        and concatenate them into a single GeoDataFrame.

        Raises:
            ValueError: If duplicate IDs are found in the combined GeoDataFrame.
        
        Returns:
            A single combined GeoDataFrame containing all processed points.
        """
        gdf_list = []
        # Process each CSV sequentially.
        for param in self.pt_csv_params:
            print(f"\r{param[0]} loaded ...             ", end='')
            gdf_list.append(self.process_csv(param))
        print()
        # Concatenate all GeoDataFrames.
        combined_gdf = pd.concat(gdf_list, ignore_index=True)
        combined_gdf["id"] = combined_gdf["id"].astype(str)
        
        # Check for duplicate IDs.
        if combined_gdf["id"].duplicated().any():
            raise ValueError("Duplicate IDs found in the combined GeoDataFrame.")
        else:
            combined_gdf = combined_gdf.set_index("id")
        
        return combined_gdf

    # def load_all_csvs(self):
    #     """
    #     Process all CSV files concurrently, verify no duplicate IDs exist across files,
    #     and concatenate them into a single GeoDataFrame.

    #     Raises:
    #         ValueError: If duplicate IDs are found in the combined GeoDataFrame.
        
    #     Returns:
    #         A single combined GeoDataFrame containing all processed points.
    #     """
    #     gdf_list = []
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
    #         # Process each CSV concurrently.
    #         futures = [executor.submit(self.process_csv, param) for param in self.pt_csv_params]
    #         for future in concurrent.futures.as_completed(futures):
    #             gdf_list.append(future.result())
        
    #     # Concatenate all GeoDataFrames.
    #     combined_gdf = pd.concat(gdf_list, ignore_index=True)
    #     combined_gdf["id"] = combined_gdf["id"].astype(str)
        
    #     # Check for duplicate IDs.
    #     if combined_gdf["id"].duplicated().any():
    #         raise ValueError("Duplicate IDs found in the combined GeoDataFrame.")
    #     else:
    #         combined_gdf.set_index("id")
        
    #     return combined_gdf

class LineLoader:
    """
    A class to load a shapefile containing LineString and MultiLineString geometries,
    break each into individual line segments using multi-threading, and extend each
    segment with specified attribute fields from the original data.
    """
    def __init__(self, shp_path, shp_param, target_crs, max_workers=8):
        """
        Initializes the LineLoader instance.
        
        Parameters:
            shp_path (str): Path to the input shapefile.
            idx_fields (list): List of attribute field names to transfer from the original 
                               line object to each generated segment.
            max_workers (int): The maximum number of threads to use for processing.
        """
        # Read the shapefile and load only the specified fields plus geometry.
        self.shp_param = shp_param
        self.gdf = gpd.read_file(shp_path, columns=shp_param + ['geometry'])
        # Ensure all required fields exist in the GeoDataFrame
        missing_fields = [field for field in shp_param if field not in self.gdf.columns]
        if missing_fields:
            raise ValueError(f"Fields {missing_fields} not found in {shp_path}")
        # If source and target CRS differ, perform coordinate transformation.
        if self.gdf.crs != target_crs:
            self.gdf = self.gdf.to_crs(target_crs)
        self.max_workers = max_workers
        self.node_dict = {}
        self.node_count = 0

    def _encode_node(self, coord, Line_Id):
        if coord not in self.node_dict:
            self.node_dict[coord] = f'{Line_Id}_{self.node_count:X}'
            self.node_count += 1
        return self.node_dict[coord]

    def _process_linestring(self, linestring, attributes):
        """
        Process a single LineString object to extract its segments.

        Parameters:
            linestring (LineString): A Shapely LineString geometry.
            attributes (dict): A dictionary of attributes to add to each segment.

        Returns:
            list: A list of dictionaries, each containing a segment's coordinate list,
                its geometry, an encoding, and additional attributes.
        """
        coords = tuple(linestring.coords)
        line_Id = attributes[self.shp_param[0]]
        
        # Precompute encoded values for all coordinates to avoid redundant calls.
        encoded_coords = [self._encode_node(coord, line_Id) for coord in coords]
        
        # Use a list comprehension with dictionary unpacking to create segments.
        segments = [
            {
                'geometry': LineString((coords[i], coords[i+1])),
                'coord_list': (coords[i], coords[i+1]),
                'encoding': (encoded_coords[i],encoded_coords[i+1]),
                **attributes
            }
            for i in range(len(coords) - 1)
        ]
        return segments

    def _process_row(self, row):
        """
        Process a single row to extract individual line segments while copying 
        the specified attributes.
        
        Parameters:
            row (namedtuple): A row from the GeoDataFrame with a 'geometry' attribute and
                              additional fields specified in idx_fields.
        
        Returns:
            list: A list of dictionaries. Each dictionary contains a segment geometry
                  and the specified attribute fields.
        """
        segments = []
        geom = row.geometry
        
        # Create a dictionary of the specified attribute fields.
        attributes = {field: getattr(row, field) for field in self.shp_param}
        
        if geom is None:
            return segments
        
        # Process a single LineString using the helper function.
        if geom.geom_type == 'LineString':
            segments.extend(self._process_linestring(geom, attributes))
        # Process a MultiLineString by iterating over its individual LineStrings.
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                segments.extend(self._process_linestring(line, attributes))
        return segments

    def load_lines(self):
        """
        Breaks down all line segments in the shapefile in a single thread and 
        extends each segment with the specified attribute fields.
        
        Returns:
            GeoDataFrame: A new GeoDataFrame where each row represents an individual
                          line segment along with the transferred attribute fields.
        """
        all_segments = []
        # Process each row sequentially
        for row in self.gdf.itertuples(index=False):
            segments = self._process_row(row)
            all_segments.extend(segments)
        
        # Create a new GeoDataFrame from the list of segment dictionaries.
        self.gdf = gpd.GeoDataFrame(all_segments, crs=self.gdf.crs)
        # reverse the sequence of line nodes. Although this dict will no longer be used, 
        self.node_dict = {v: k for k, v in self.node_dict.items()}

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