import networkx as nx
import pandas as pd
import multiprocessing as mp

numeric_val = 6

def save_edgelist(filepath: str, df: pd.DataFrame):
    with open(filepath, 'w') as f:
        f.write(''.join(f"{src} {end} {length}\n" for src, end, length in df[['src', 'end', 'weight']].values))

def save_paths(filepath: str, paths: list, od_df: pd.DataFrame, index: bool = False):
    """
    Save a DataFrame to a CSV file with UTF-8 encoding.
    
    :param filepath: The path where the file will be saved.
    :param paths: A list containing distance and path information.
    :param od_file: The original OD DataFrame.
    :param index: Whether to include the index in the saved file (default: False).
    """
    # Ensure paths is properly formatted as a DataFrame
    paths_df = pd.DataFrame(paths, columns=['dist', 'path'])

    # Concatenate od_file (left) and paths_df (right)
    paths_df = pd.concat([od_df.reset_index(drop=True), paths_df.reset_index(drop=True)], axis=1)

    try:
        paths_df.to_csv(filepath, encoding='utf-8-sig', index=index)
        print(f"Data successfully saved to {filepath}")
    except Exception as e:
        raise ValueError(f"Error saving DataFrame: {e}")

# Read file with space as row separator and process chunks
def _pd_edgelist(filename, chunksize=1e5):
    chunk_list = []
    for chunk in pd.read_csv(filename, sep=" ", names=['src', 'end', 'weight'], chunksize=chunksize, engine='c'):
        chunk_list.append(chunk)  # Store each chunk
    return pd.concat(chunk_list, ignore_index=True)  # Merge all chunks

def pd_edgelist(filename):
    return pd.read_csv(filename, sep=" ", names=['src', 'end', 'weight'], engine='pyarrow')

def pd_intersect_reassign(df, target_nodes, target_field, cost):
    # Boolean mask: Select rows where at least one column is in target_nodes
    mask = df[target_field].isin(target_nodes)
    # Update weights at selected rows
    df.loc[mask, 'weight'] = cost
    return df

def pd_avg_weight(df, avg_speed):
    if 'weight' in df.columns:  # Ensure 'weight' is a column
        df['weight'] = df['weight'].astype(float) / avg_speed
    else:
        raise KeyError("Column 'weight' not found in DataFrame")
    return df

def pd_od_readin(filename):
    df = pd.read_csv(filename, header=0, engine='python', dtype=str)
    # Check if the DataFrame has exactly two columns
    if df.shape[1] != 2:
        raise ValueError(f"Expected 2 columns ('O' and 'D'), but got {df.shape[1]} columns.")
    df.columns = ['O', 'D']
    return df

def _single_source_path_lengths(G: nx.MultiDiGraph, ori: str, dests: set, weight: str,
                                cutoff_threshod: float, self_weight: float = 0):
    if not  G.has_node(ori):
        return('Not Connected to Road Network')
    # Note: Dijkstra's algorithm is not guaranteed to work if edge weights are negative or are floating point numbers (overflows and roundoff errors can cause problems).
    reachable_nodes = nx.single_source_dijkstra_path_length(G, ori, cutoff=cutoff_threshod, weight=weight)
    # 取交集并生成新的字典, 保存在 result 中
    return({k: (self_weight if k == ori else round(reachable_nodes[k], numeric_val)) for k in reachable_nodes.keys() & dests})

def source_shortest_path_length(
        G: nx.MultiDiGraph, 
        origs: set, 
        dests: set, 
        weight: str,
        cutoff_threshold: float, 
        self_weight: float = 0,
        cpus: int | None = 1):
    
    # determine how many cpu cores to use
    if cpus is None:
        cpus = mp.cpu_count()
    cpus = min(cpus, mp.cpu_count())

    # if single-threading, calculate each shortest path one at a time
    if cpus <= 1:
        path_lengths = {o: _single_source_path_lengths(G, o, dests, weight, cutoff_threshold, self_weight) for o in origs}
    # if multi-threading, calculate shortest paths in parallel
    else:
        args = [(G, o, dests, weight, cutoff_threshold, self_weight) for o in origs]
        with mp.get_context().Pool(cpus) as pool:
            results = pool.starmap(_single_source_path_lengths, args)
        # Reconstruct dictionary from results
        path_lengths = {o: res for o, res in zip(origs, results)}
    
    return path_lengths

def _single_shortest_path(
    G: nx.MultiDiGraph,
    orig: str,
    dest: str,
    weight: str,
) -> list[int] | None:
    """
    A copied function from https://github.com/gboeing/osmnx/blob/main/osmnx/routing.py

    Solve the shortest path from an origin node to a destination node.

    This function uses Dijkstra's algorithm. It is a convenience wrapper
    around `networkx.shortest_path`, with exception handling for unsolvable
    paths. If the path is unsolvable, it returns None.
    Note: Dijkstra's algorithm is not guaranteed to work if edge weights are negative or are floating point numbers (overflows and roundoff errors can cause problems).

    Parameters
    ----------
    G
        Input graph.
    orig
        Origin node ID.
    dest
        Destination node ID.
    weight
        Edge attribute to minimize when solving shortest path.

    Returns
    -------
    distance, path
        The node IDs constituting the shortest path and corresponding cost in network
    """
    try:
        #return tuple(nx.single_source_dijkstra(G, orig, target=dest, cutoff=None, weight=weight))
        #return tuple(nx.single_source_bellman_ford(G, orig, target=dest, weight='weight'))
        # bidirectional_dijkstra is twice faster (for no surprise) and ensures absolute return of input pairs. 
        # I am not sure what will happen if path between origin and destination doesn't exist.
        return tuple(nx.bidirectional_dijkstra(G, orig, target=dest, weight=weight))
    except nx.exception.NetworkXNoPath:  # pragma: no cover
        msg = f"Cannot solve path from {orig} to {dest}"
        print(msg)
        return None, None

def shortest_path(G: nx.MultiDiGraph, 
                  od_df: pd.DataFrame,
                  weight: str = "length",
                  cpus: int | None = 1,
                ) -> list[int] | None | list[list[int] | None]:
    """
    A copied function from https://github.com/gboeing/osmnx/blob/main/osmnx/routing.py

    Solve shortest path from origin node(s) to destination node(s).

    Uses Dijkstra's algorithm. If `orig` and `dest` are single node IDs, this
    will return a list of the nodes constituting the shortest path between
    them. If `orig` and `dest` are lists of node IDs, this will return a list
    of lists of the nodes constituting the shortest path between each
    origin-destination pair. If a path cannot be solved, this will return None
    for that path. You can parallelize solving multiple paths with the `cpus`
    parameter, but be careful to not exceed your available RAM.

    Parameters
    ----------
    G
        Input graph.
    od_df
        OD dataframe for distance & path query
    weight
        Edge attribute to minimize when solving shortest path.
    cpus
        How many CPU cores to use if multiprocessing. If None, use all
        available. If you are multiprocessing, make sure you protect your
        entry point: see the Python docs for details.

    Returns
    -------
    path
        The node IDs constituting the shortest path, or, if `orig` and `dest`
        are both iterable, then a list of such paths.
    """

    # determine how many cpu cores to use
    if cpus is None:
        cpus = mp.cpu_count()
    cpus = min(cpus, mp.cpu_count())

    # if single-threading, calculate each shortest path one at a time
    if cpus <= 1:
        paths = [_single_shortest_path(G, o, d, weight) for o, d in od_df[['O', 'D']].values]

    # if multi-threading, calculate shortest paths in parallel
    else:
        # with mp.get_context().Pool(cpus) as pool:
        #     paths = pool.starmap_async(_single_shortest_path, args).get()
        args = [(G, o, d, weight) for o, d in od_df[['O', 'D']].values]
        with mp.get_context().Pool(cpus) as pool:
            paths = pool.starmap(_single_shortest_path, args)

    return paths

def open_edgelist(file_path: list, is_direct: bool):
    if is_direct:
        G = nx.read_weighted_edgelist(file_path, create_using=nx.MultiDiGraph, nodetype=str)
    else:
        G = nx.read_weighted_edgelist(file_path, create_using=nx.Graph, nodetype=str)
    return G

def create_network_graph(edges: list, is_direct: bool):
    """
    Generate a NetworkX multi-directed graph from the projected point edges.
    
    This function calls create_edges() to get the edge list and then
    builds a directed graph (nx.MultiDiGraph) where each edge has an attribute 'weight'.
    
    Parameters:
      pt_pdf (DataFrame): Must contain "line_id", "pt_id", and "prj_length".
      line_gdf (GeoDataFrame): Must contain "line_id", "geometry", and "encoding".
    
    Returns:
      A networkX DiGraph with edges added.
    """
    if is_direct:
        G = nx.MultiDiGraph()
    else:
        G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return G