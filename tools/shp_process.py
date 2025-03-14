import pandas as pd
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString

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
    def __init__(self, shp_path, shp_param, target_crs, prefix):
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
        self.prefix = prefix
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

    def _encode_node(self, coord: tuple[float, float]) -> str:
        """
        Encodes a coordinate (tuple of floats) as a unique node identifier.
        
        If the coordinate has not been previously encoded, it is assigned a new 
        identifier in the format 'L' followed by a hexadecimal representation 
        of the current node count. The node count is incremented with each new
        unique coordinate.

        Parameters:
        - coord: A tuple representing the coordinate (x, y) of the node.

        Returns:
        - A unique string identifier for the node.
        """
        if coord not in self.node_dict:
            self.node_dict[coord] = f'{self.prefix}_{self.node_count:X}'
            self.node_count += 1
        return self.node_dict[coord]

    def _process_linestring(self, linestring: LineString, attributes: tuple) -> list:
        """
        Process a single LineString object to extract its segments.

        Parameters:
            linestring (LineString): A Shapely LineString geometry.
            attributes (tuple): A tuple of attributes to add to each segment.

        Returns:
            list: A list of dictionaries, each containing a segment's coordinate list,
                its geometry, an encoding, and additional attributes.
        """
        coords = tuple(linestring.coords)
        
        # Precompute encoded values for all coordinates to avoid redundant calls.
        encoded_coords = [self._encode_node(coord) for coord in coords]
        
        # Use a list comprehension to create segments.
        ### 'geometry', 'scr_encode', 'end_encode'
        segments = [
            (LineString((coords[i], coords[i+1])), encoded_coords[i], encoded_coords[i+1]) + attributes
            for i in range(len(coords) - 1)
        ]
        return segments

    def _process_row(self, row: gpd.GeoSeries) -> list:
            """
            Processes a row from a GeoDataFrame and returns a list of segments based on the geometry type.
            
            This function processes a single row from a GeoDataFrame. It first extracts the geometry and a list of
            specified attributes. Then it processes the geometry:
            - For a LineString, it calls the `_process_linestring` helper function.
            - For a MultiLineString, it iterates through the individual LineStrings and processes each one.
            
            Parameters:
            - row: A GeoSeries representing a single row in the GeoDataFrame containing geometry and attributes.

            Returns:
            - A list of segments obtained by processing the geometry in the row.
            """
            segments = []
            geom = row.geometry
            
            # Create a list of the specified attribute fields in the given order.
            attributes = tuple(getattr(row, field) for field in self.shp_param)
            
            if geom is None:
                return segments
            
            # Process a single LineString using the helper function.
            if isinstance(geom, LineString):
                segments.extend(self._process_linestring(geom, attributes))
            # Process a MultiLineString by iterating over its individual LineStrings.
            elif isinstance(geom, MultiLineString):
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
        self.gdf = gpd.GeoDataFrame(all_segments, crs=self.gdf.crs,\
                                    columns=['geometry','scr_encode','end_encode']+self.shp_param)
        # reverse the sequence of line nodes. Although this dict will no longer be used, 
        self.node_dict = {v: k for k, v in self.node_dict.items()}

def nearby_lines(lines_geom: gpd.GeoSeries, points_geom: gpd.GeoSeries, buffer_distance: float) -> tuple:
    """
    Identifies nearby lines based on the proximity of points within a specified buffer distance.

    Parameters:
    - lines_geom (gpd.GeoSeries): A GeoSeries containing geometries of lines (e.g., LineStrings).
    - points_geom (gpd.GeoSeries): A GeoSeries containing point geometries (e.g., Points).
    - buffer_distance (float): The distance (in the same coordinate reference system as the geometries) to buffer each point, creating a search region.

    Returns:
    - tuple[np.ndarray, np.ndarray]:
        - np.ndarray: The indices of the lines that intersect the union of all point buffers.
        - np.ndarray: A 3D numpy array of coordinates (x, y) of the intersecting lines, reshaped to form a 2x2 array for each line segment.
    """
    # Create buffers for each point and then compute the union of all buffers.
    union_buffer = points_geom.buffer(buffer_distance).unary_union
    # Use the union buffer to find all line features that intersect it.
    nearby_lines_index = lines_geom[lines_geom.intersects(union_buffer)].index.to_numpy()
    return nearby_lines_index, lines_geom.iloc[nearby_lines_index].get_coordinates().to_numpy().reshape(-1,2,2)

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

def create_edges(prj_gdf: gpd.GeoDataFrame, pt_gdf: gpd.GeoDataFrame, line_gdf: gpd.GeoDataFrame, net_prefix) -> gpd.GeoDataFrame:
    """
    Creates edges by processing and merging project, point, and line GeoDataFrames to form network connections.

    This function performs the following tasks:
    - Validates required columns in input GeoDataFrames.
    - Sorts and processes project GeoDataFrame to create connection edges.
    - Creates connecting lines for the network.
    - Handles the creation of start and end lines for each project.
    - Merges all processed data into a final network GeoDataFrame.

    Parameters:
    - prj_gdf: A GeoDataFrame containing project-related data with columns 'line_id', 'pt_id', and 'prj_length'.
    - pt_gdf: A GeoDataFrame containing point-related data with geometry.
    - line_gdf: A GeoDataFrame containing line-related data, must have 'scr_encode' and 'end_encode' columns.

    Returns:
    - A GeoDataFrame representing the created edges (lines) with calculated lengths.
    """
    # Validate prj_gdf columns
    required_cols = {"line_id", "pt_id", "prj_length"}
    if not required_cols.issubset(prj_gdf.columns):
        raise KeyError(f"prj_gdf must contain columns: {required_cols}")
   
    # Validate that line_gdf has the required columns
    required_cols = {"scr_encode", "end_encode"}
    if not required_cols.issubset(line_gdf.columns):
        raise KeyError(f"line_gdf must contain columns: {required_cols}")

    print("Pre-processing Dataframe ...", end='')
    # Sort points within each line_id by prj_length
    prj_gdf = prj_gdf.sort_values(by=['line_id', 'prj_length'])
    prj_gdf['current_node'] = net_prefix + prj_gdf['line_id'].astype(str) + '-' + prj_gdf['pt_id'].astype(str)

    pt_gdf.rename(columns={"geometry": "pt_geom"}, inplace=True)
    prj_gdf.rename(columns={"geometry": "prj_geom"}, inplace=True)
    cols_rmv = pt_gdf.columns.to_list() + prj_gdf.columns.to_list()

    merged = prj_gdf.merge(pt_gdf, left_on='pt_id', right_index=True)
    merged = merged.merge(line_gdf, left_on='line_id', right_index=True)

    print("\rConstructing projected lines ...       ", end='')
    prj_lines_gdf = merged.copy()
    prj_lines_gdf['end_encode'] = prj_lines_gdf['current_node']
    prj_lines_gdf['scr_encode'] = prj_lines_gdf['current_node'].shift(1)
    prj_lines_gdf['prev_prj_geom'] = prj_lines_gdf['prj_geom'].shift(1)
    prj_lines_gdf.loc[prj_lines_gdf.index[1:], 'geometry'] = [
        LineString([prev, curr]) for prev, curr in prj_lines_gdf[['prev_prj_geom', 'prj_geom']].values[1:]
    ]
    prj_lines_gdf = prj_lines_gdf[prj_lines_gdf['line_id'].duplicated(keep='first')]
    prj_lines_gdf.reset_index(drop=True, inplace=True)
    prj_lines_gdf.drop(columns=cols_rmv + ['prev_prj_geom'], inplace=True)

    print("\rConstructing connecting lines ...       ", end='')
    to_network_gdf = merged.copy()
    to_network_gdf['scr_encode'] = to_network_gdf['pt_id']
    to_network_gdf['end_encode'] = to_network_gdf['current_node']
    to_network_gdf['geometry'] = [LineString([pt, prj]) for pt, prj in to_network_gdf[['pt_geom', 'prj_geom']].values]
    to_network_gdf.drop(columns=cols_rmv, inplace=True)

    print("\rConstructing start lines ...           ", end='')
    first_gdf = merged.copy().drop_duplicates(subset='line_id', keep='first')
    modified_line_ids = first_gdf['line_id'].values
    first_gdf['end_encode'] = first_gdf['current_node']
    first_gdf['geometry'] = [LineString([geom.coords[0], prj]) for geom, prj in first_gdf[['geometry', 'prj_geom']].values]
    first_gdf.drop(columns=cols_rmv, inplace=True)

    print("\rConstructing end lines ...             ", end='')
    last_gdf = merged.copy().drop_duplicates(subset='line_id', keep='last')
    last_gdf['scr_encode'] = last_gdf['current_node']
    last_gdf['geometry'] = [LineString([prj, geom.coords[1]]) for geom, prj in last_gdf[['geometry', 'prj_geom']].values]
    last_gdf.drop(columns=cols_rmv, inplace=True)

    print("\rIntegrating lines ...                   ", end='')
    # Drop the original rows that were modified
    line_gdf = line_gdf.drop(index=modified_line_ids)
    merged = pd.concat([line_gdf, prj_lines_gdf, to_network_gdf, first_gdf, last_gdf], ignore_index=True)
    line_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=line_gdf.crs)

    # Calculate lengths
    line_gdf['length'] = line_gdf.geometry.length

    print("\rEdges Created !!!                      ", end='\n')
    return line_gdf

def create_edgelist(lines_attr, direction_field=None):
    """
    Convert a DataFrame of edges into a tuple of tuples using pandas functions.
    
    Parameters:
    - lines (pd.DataFrame): DataFrame with columns 'scr_encode', 'end_encode', 'length',
                            and optionally 'Direction'.
    
    Returns:
    - tuple: Tuple of tuples, each representing an edge as (source, target, weight).
    """    
    # Validate that line_gdf has the required columns
    required_cols = {"scr_encode", "end_encode", "length"}
    if not required_cols.issubset(lines_attr.columns):
        raise KeyError(f"lines must contain columns: {required_cols}")

    # Check if 'Direction' column exists
    if direction_field and direction_field in lines_attr.columns:
        # Filter non-directional edges (Direction == 0)
        non_directional = lines_attr[lines_attr['Direction'] == 0]
        # Filter directional edges (Direction == 1)
        directional = lines_attr[lines_attr['Direction'] == 1]
        
        # Create swapped edges for non-directional rows
        swapped = non_directional[['end_encode', 'scr_encode', 'length']].rename(
            columns={'end_encode': 'scr_encode', 'scr_encode': 'end_encode'}
        )
        
        # Combine all edges: original non-directional, swapped, and directional
        lines_attr = pd.concat([non_directional, swapped, directional], ignore_index=True)    
    
    # Convert to tuple of tuples
    edges = tuple(lines_attr[['scr_encode', 'end_encode', 'length']].itertuples(index=False, name=None))
    
    return edges

if __name__ == '__main__':
    # print("Reading Points ...")
    # prj_gdf = gpd.read_file('prj_pts.shp')
    # pt_gdf = gpd.read_file('points.shp').set_index("id")
    print("Reading Lines ...")
    line_gdf = gpd.read_file('topo_walk_lines.shp')
    # print("Creating Edges ...")
    # # Call create_edges to generate edges
    # line_gdf = create_edges(prj_gdf, pt_gdf, line_gdf)
    # line_gdf.to_file("lines_mod.shp", driver="ESRI Shapefile", encoding='utf-8')
    edges = create_edgelist(line_gdf[['scr_encode', 'end_encode', 'length','Direction']], 'Direction')
    print(edges[:10])