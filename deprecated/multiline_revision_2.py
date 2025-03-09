import geopandas as gpd
from shapely.geometry import LineString, MultiLineString

def chain_segments(segments):
    """
    Attempts to chain a list of LineString segments following these rules:
      1. Start with a segment whose start node is unique.
      2. If exactly one unused segment connects (by sharing its start or end)
         to the current chain’s endpoint, append it (reverse its orientation if needed).
      3. If more than one segment is found, replace the endpoint of the current
         segment with the endpoint of one candidate that shares the common node.
    If all segments can be chained (used exactly once), returns the merged LineString;
    otherwise, returns None.
    """
    # Wrap segments with a flag to indicate if they have been used
    segs = [{'geom': seg, 'used': False} for seg in segments]

    # Get start and end coordinates for each segment as tuples
    starts = [tuple(seg['geom'].coords[0]) for seg in segs]
    ends = [tuple(seg['geom'].coords[-1]) for seg in segs]

    # Identify a candidate starting segment:
    # Use one whose start appears only once and is not an endpoint of any segment.
    start_candidates = []
    for i, start in enumerate(starts):
        if starts.count(start) == 1 and ends.count(start) == 0:
            start_candidates.append(i)
    # If no candidate is found, choose the first segment arbitrarily.
    start_idx = start_candidates[0] if start_candidates else 0

    # Initialize the chain with the starting segment.
    segs[start_idx]['used'] = True
    chain = [segs[start_idx]['geom']]
    current_endpoint = tuple(chain[-1].coords[-1])

    # Attempt to chain remaining segments.
    while True:
        # Find candidate segments that have a connection at the current endpoint.
        candidates = []
        for i, seg in enumerate(segs):
            if seg['used']:
                continue
            coords = list(seg['geom'].coords)
            if tuple(coords[0]) == current_endpoint or tuple(coords[-1]) == current_endpoint:
                candidates.append((i, seg['geom']))
        # If no candidate is found, break out.
        if not candidates:
            break

        if len(candidates) == 1:
            # Only one connecting segment: use it and reverse its orientation if needed.
            i, candidate = candidates[0]
            coords = list(candidate.coords)
            if tuple(coords[0]) != current_endpoint:
                coords.reverse()
                candidate = LineString(coords)
            chain.append(candidate)
            segs[i]['used'] = True
            current_endpoint = tuple(candidate.coords[-1])
        else:
            # More than one candidate connection:
            # According to the rule, replace the endpoint of the current segment with the
            # endpoint of one candidate that shares its start with the current endpoint.
            selected = None
            for i, candidate in candidates:
                coords = list(candidate.coords)
                if tuple(coords[0]) == current_endpoint:
                    selected = (i, candidate)
                    break
            if not selected:
                # If no candidate has its first coordinate equal to the current endpoint,
                # choose the first candidate and reverse its coordinates.
                i, candidate = candidates[0]
                coords = list(candidate.coords)
                coords.reverse()
                candidate = LineString(coords)
                selected = (i, candidate)
            i, selected_candidate = selected

            # Modify the last segment in the chain by replacing its end with the selected
            # candidate's endpoint.
            last_seg_coords = list(chain[-1].coords)
            new_endpoint = selected_candidate.coords[-1]
            last_seg_coords[-1] = new_endpoint
            chain[-1] = LineString(last_seg_coords)
            segs[i]['used'] = True
            current_endpoint = tuple(new_endpoint)

    # Check if all segments have been used to form a continuous chain.
    all_used = all(seg['used'] for seg in segs)
    if all_used:
        # Merge the chain into a single LineString (avoiding duplicate nodes).
        merged_coords = list(chain[0].coords)
        for seg in chain[1:]:
            seg_coords = list(seg.coords)
            merged_coords.extend(seg_coords[1:])
        return LineString(merged_coords)
    else:
        return None

def process_shapefile(input_shp, output_passed_shp, output_rejected_shp):
    """
    Reads an input shapefile, processes each multiline feature, and then saves features that
    can be chained into a continuous line (passed) separately from those that cannot (rejected).
    """
    # Read the shapefile
    gdf = gpd.read_file(input_shp)
    passed_features = []
    rejected_features = []
    max_idx = gdf.shape[0]
    # Process each feature
    for idx, row in gdf.iterrows():
        print(f"\rProcessing feature {idx+1}/{max_idx}", end='')
        geom = row.geometry
        if geom is None:
            rejected_features.append(row)
            continue
        # Extract segments from MultiLineString or use the LineString as-is.
        if geom.type == 'MultiLineString':
            segments = list(geom.geoms)
        elif geom.type == 'LineString':
            segments = [geom]
        else:
            # Unsupported geometry type; mark as rejected.
            rejected_features.append(row)
            continue

        # Attempt to chain the segments.
        merged_line = chain_segments(segments)
        if merged_line is not None:
            row.geometry = merged_line
            passed_features.append(row)
        else:
            rejected_features.append(row)

    # Create GeoDataFrames for passed and rejected features.
    passed_gdf = gpd.GeoDataFrame(passed_features, crs=gdf.crs)
    rejected_gdf = gpd.GeoDataFrame(rejected_features, crs=gdf.crs)

    # Save the results to new shapefiles.
    passed_gdf.to_file(output_passed_shp)
    rejected_gdf.to_file(output_rejected_shp)
    print(f"Processing complete. Passed: {len(passed_gdf)}, Rejected: {len(rejected_gdf)}")

if __name__ == '__main__':
    process_shapefile("disconnected_lines.shp", "output_passed_shp", "output_rejected_shp")
