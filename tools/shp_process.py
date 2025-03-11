import pandas as pd
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString

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
        self.gdf = None

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
        self.gdf = []
        # Process each CSV sequentially.
        for param in self.pt_csv_params:
            print(f"\r{param[0]} loaded ...             ", end='')
            self.gdf.append(self.process_csv(param))
        print()
        # Concatenate all GeoDataFrames.
        self.gdf = pd.concat(self.gdf, ignore_index=True)
        self.gdf["id"] = self.gdf["id"].astype(str)
        
        # Check for duplicate IDs.
        if self.gdf["id"].duplicated().any():
            raise ValueError("Duplicate IDs found in the combined GeoDataFrame.")
        else:
            self.gdf.set_index("id", inplace=True)

class LineLoader:
    """
    A class to load a shapefile containing LineString and MultiLineString geometries,
    break each into individual line segments using multi-threading, and extend each
    segment with specified attribute fields from the original data.
    """
    def __init__(self, shp_path, shp_param, target_crs):
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
        self.node_dict = {}
        self.node_count = 0

    def _encode_node(self, coord):
        if coord not in self.node_dict:
            self.node_dict[coord] = f'L{self.node_count:X}'
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
        
        # Precompute encoded values for all coordinates to avoid redundant calls.
        encoded_coords = [self._encode_node(coord) for coord in coords]
        
        # Use a list comprehension with dictionary unpacking to create segments.
        segments = [
            {
                'geometry': LineString((coords[i], coords[i+1])),
                'scr_encode': encoded_coords[i],
                'end_encode': encoded_coords[i+1],
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

def nearby_lines(lines_geom: gpd.GeoSeries, points_geom: gpd.GeoSeries, buffer_distance: float):
    """
    Determines which line geometries intersect the union of buffered point geometries.

    This function creates a buffer around each point in the provided `points_geom` using the specified
    `buffer_distance`. It then computes the union of all these buffers to form a single geometry. Finally,
    it checks which line features in `lines_geom` intersect this unioned buffer and returns their indices.

    Parameters:
        lines_geom (GeoSeries): A GeoSeries containing the line geometries. This could also be the 
                                geometry column from a GeoDataFrame of lines.
        points_geom (GeoSeries): A GeoSeries containing the point geometries. This could also be the 
                                 geometry column from a GeoDataFrame of points.
        buffer_distance (float): The distance (in the same units as the CRS of the geometries) used to 
                                 buffer each point.

    Returns:
        tuple:
            status (bool): True if one or more line features intersect the union of the point buffers,
                           False otherwise.
            nearby_lines_index (numpy.ndarray): A NumPy array containing the indices of the line features
                                                  from `lines_geom` that intersect with the unioned buffer.

    Example:
        >>> status, indices = get_nearby(lines.geometry, points.geometry, 100)
        >>> print(status)  # True if at least one line intersects the buffered area
        >>> print(indices) # Array of indices of intersecting line features

    Notes:
        - Ensure that both `lines_geom` and `points_geom` have the same Coordinate Reference System (CRS)
          to guarantee accurate spatial operations.
        - The function uses `unary_union` to efficiently combine all point buffers into one geometry.
    """
    # Create buffers for each point and then compute the union of all buffers.
    union_buffer = points_geom.buffer(buffer_distance).unary_union

    # Use the union buffer to find all line features that intersect it.
    nearby_lines_index = lines_geom[lines_geom.intersects(union_buffer)].index.to_numpy()

    # Determine the status based on whether any intersections were found.
    status = nearby_lines_index.size > 0

    return status, nearby_lines_index, lines_geom.iloc[nearby_lines_index].get_coordinates().to_numpy().reshape(-1,2,2)

def arr2gdf(attrs_arr, x_arr, y_arr, col_names: list, input_crs: CRS):
    """
    Creates a GeoDataFrame from attribute data and saves both the project points and lines to files.
    
    Parameters:
        attrs_lst (list): List of attribute arrays.
        idxs_lst (list): List of index arrays corresponding to points and lines.
        points (DataFrame): DataFrame containing point features.
        lines (GeoDataFrame): GeoDataFrame containing line features.
        target_crs (str or dict): The coordinate reference system.
        prj_filename (str): Filename for the project points shapefile.
        lines_filename (str): Filename for the lines shapefile.
    """
    # Create GeoDataFrame for project points using GeoPandas
    gdf = gpd.GeoDataFrame(
        attrs_arr,
        geometry=gpd.points_from_xy(x_arr, y_arr),
        columns=col_names,
        crs=input_crs
    )
    return gdf

def create_edges(prj_gdf: gpd.GeoDataFrame, pt_gdf: gpd.GeoDataFrame, line_gdf: gpd.GeoDataFrame):
    # Validate prj_gdf columns
    required_cols = {"line_id", "pt_id", "prj_length"}
    if not required_cols.issubset(prj_gdf.columns):
        raise KeyError(f"prj_gdf must contain columns: {required_cols}")
   
    # Validate that line_gdf has the required columns
    required_cols = {"scr_encode", "end_encode"}
    if not required_cols.issubset(line_gdf.columns):
        raise KeyError(f"line_gdf must contain columns: {required_cols}")

    modified_edge_idx = set()
    new_rows = []
    # Get the indices of the columns to be modified (assumed 'scr_encode', 'end_encode', 'geometry')
    scr_encode_idx = line_gdf.columns.get_loc('scr_encode')
    end_encode_idx = line_gdf.columns.get_loc('end_encode')
    geometry_idx = line_gdf.columns.get_loc('geometry')
    
    # Sort points within each line_id by prj_length
    prj_gdf = prj_gdf.sort_values(by=['line_id', 'prj_length'])

    # Iterate through each line_id group
    for line_id, group in prj_gdf.groupby("line_id"):
        line_row = line_gdf.iloc[line_id]  # Direct row access by index
        prev_geom, end_geom = tuple(line_row.geometry.coords)
        base_row = line_row.to_numpy()  # Base row as list of column values
        prev_node = base_row[scr_encode_idx] 
        
        # Pre-calculate the number of new rows to create for this group
        num_new_rows = group.shape[0] * 3 + 1  # 3 new rows per point + 1 final row
        # Pre-allocate the space for new rows, duplicating base_row
        segment_rows = base_row.repeat(num_new_rows, axis=0).reshape(-1, num_new_rows).T

        # Iterate over the points in the group and generate new rows
        idx = 0  # Index for assigning to pre-allocated segment_rows
        for _, row in group.iterrows():
            pt_id = row['pt_id']
            current_node = f"{line_id}-{pt_id}"
            current_geom = row.geometry

            # Assign the values to the pre-allocated rows
            segment_rows[idx, scr_encode_idx] = prev_node
            segment_rows[idx, end_encode_idx] = current_node
            segment_rows[idx, geometry_idx] = LineString([prev_geom, current_geom])
            idx += 1

            segment_rows[idx, scr_encode_idx] = pt_id
            segment_rows[idx, end_encode_idx] = current_node
            segment_rows[idx, geometry_idx] = LineString([pt_gdf.loc[pt_id].geometry, current_geom])
            idx += 1

            segment_rows[idx, scr_encode_idx] = current_node
            segment_rows[idx, end_encode_idx] = pt_id
            segment_rows[idx, geometry_idx] = LineString([current_geom, pt_gdf.loc[pt_id].geometry])
            idx += 1
            
            # Update previous node and geometry for the next iteration
            prev_node = current_node
            prev_geom = current_geom

        # Add the final row for the last projected point
        segment_rows[idx, scr_encode_idx] = prev_node
        segment_rows[idx, geometry_idx] = LineString([prev_geom, end_geom])

        # Add the generated rows to new_rows
        new_rows.extend(segment_rows)

    # Drop modified rows based on their indices
    line_gdf.drop(index=modified_edge_idx, inplace=True)

    # Convert the list of rows into a GeoDataFrame
    new_gdf = gpd.GeoDataFrame(new_rows, columns=line_gdf.columns, geometry='geometry', crs=line_gdf.crs)

    # Append the new rows to the original GeoDataFrame
    line_gdf = pd.concat([line_gdf, new_gdf], ignore_index=True)
    line_gdf['length'] = line_gdf.geometry.length

    return line_gdf

if __name__ == '__main__':
    print("Reading Points ...")
    prj_gdf = gpd.read_file('prj_pts.shp')
    pt_gdf = gpd.read_file('points.shp').set_index("id")
    print("Reading Lines ...")
    line_gdf = gpd.read_file('lines.shp')
    print("Creating Edges ...")
    # Call create_edges to generate edges
    edges, line_gdf = create_edges(prj_gdf, pt_gdf, line_gdf)
    line_gdf.to_file("lines_tmp.shp", driver="ESRI Shapefile", encoding='utf-8')