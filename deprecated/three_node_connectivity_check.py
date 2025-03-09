import geopandas as gpd
from shapely.geometry import MultiLineString, LineString

def count_node_connections(geom):
    """
    For a given geometry (MultiLineString or LineString),
    count the connections at each node by iterating over each segment.
    A segment is defined by two consecutive points.
    """
    node_count = {}
    
    # Get a list of line components (even if the geometry is a single LineString)
    if geom.geom_type == 'MultiLineString':
        lines = list(geom.geoms)
    elif geom.geom_type == 'LineString':
        lines = [geom]
    else:
        # If geometry is not a line, return an empty dict.
        return node_count

    for line in lines:
        coords = list(line.coords)
        # Iterate over each segment (pair of consecutive points)
        for i in range(len(coords) - 1):
            pt1 = coords[i]
            pt2 = coords[i + 1]
            # Count each endpoint occurrence.
            node_count[pt1] = node_count.get(pt1, 0) + 1
            node_count[pt2] = node_count.get(pt2, 0) + 1
            
    return node_count

def assemble_line(geom):
    """
    Assemble a continuous line from a MultiLineString (or LineString) by
    starting from the first segment's first point and following segments
    until the entire feature is processed. If the next segment is found to
    have its end point matching the current endpoint, the segment is reversed.
    """
    # If the geometry is already a LineString, return it as is.
    if geom.geom_type == 'LineString':
        return geom

    # Convert each component of the MultiLineString into a list of coordinates.
    segments = [list(line.coords) for line in geom.geoms]

    # Start with the first segment.
    assembled = segments.pop(0)

    # Greedily attach segments that connect to the current endpoint.
    changed = True
    while segments and changed:
        changed = False
        for i, seg in enumerate(segments):
            # If the segment's first coordinate matches the current endpoint,
            # append it directly (skipping the first coordinate to avoid duplicates).
            if seg[0] == assembled[-1]:
                assembled.extend(seg[1:])
                segments.pop(i)
                changed = True
                break
            # If the segment's last coordinate matches the current endpoint,
            # reverse the segment and then append it.
            elif seg[-1] == assembled[-1]:
                seg_reversed = seg[::-1]
                assembled.extend(seg_reversed[1:])
                segments.pop(i)
                changed = True
                break

    return LineString(assembled)

def main():
    # Define input and output shapefile paths (adjust as needed)
    input_shapefile = "disconnected_lines.shp"
    passed_shapefile = "passed.shp"
    rejected_shapefile = "rejected.shp"
    
    # Read the input shapefile using GeoPandas
    gdf = gpd.read_file(input_shapefile)
    
    # Lists to store indices for passed and rejected features
    passed_features = []
    rejected_features = []
    
    # Iterate over each feature in the shapefile with a progress print-out.
    max_idx = gdf.shape[0]
    for idx, row in gdf.iterrows():
        print(f"\rProcessing feature {idx+1}/{max_idx}", end='')
        geom = row.geometry
        node_counts = count_node_connections(geom)
        # Check if the maximum node connection count is less than 3.
        # (i.e. no node connects to 3 or more segments)
        if max(node_counts.values()) < 3:
            passed_features.append(idx)
        else:
            rejected_features.append(idx)
    print()
    
    # Process passed features: assemble the multi-lines into a continuous line.
    for idx in passed_features:
        geom = gdf.at[idx, 'geometry']
        # Only process if geometry is a MultiLineString.
        if geom.geom_type == 'MultiLineString':
            new_geom = assemble_line(geom)
            gdf.at[idx, 'geometry'] = new_geom

    # Extract passed and rejected GeoDataFrames
    passed_gdf = gdf.loc[passed_features]
    rejected_gdf = gdf.loc[rejected_features]
    
    # Write out the passed features (now with assembled continuous lines) to a new shapefile.
    if not passed_gdf.empty:
        passed_gdf.to_file(passed_shapefile)
        print(f"Extracted and processed {len(passed_gdf)} feature(s) to {passed_shapefile}")
    else:
        print("No features passed the condition.")
    
    # Write out the rejected features to a new shapefile.
    if not rejected_gdf.empty:
        rejected_gdf.to_file(rejected_shapefile)
        print(f"Extracted {len(rejected_gdf)} feature(s) to {rejected_shapefile}")
    else:
        print("No features rejected the condition.")

if __name__ == "__main__":
    main()
