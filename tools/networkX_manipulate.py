import networkx as nx
import pandas as pd

def save_edgelist(filepath: str, df: pd.DataFrame):
    with open(filepath, 'w') as f:
        f.write(''.join(f"{src} {end} {length}\n" for src, end, length in df[['src', 'end', 'weight']].values))

# Read file with space as row separator and process chunks
def pd_edgelist(filename, chunksize=1e5):
    chunk_list = []
    for chunk in pd.read_csv(filename, sep=" ", names=['src', 'end', 'weight'], chunksize=chunksize, engine='c'):
        chunk_list.append(chunk)  # Store each chunk
    return pd.concat(chunk_list, ignore_index=True)  # Merge all chunks

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

def create_network_graph(edges: list, is_direct: bool):
    """
    Generate a NetworkX directed graph from the projected point edges.
    
    This function calls create_edges() to get the edge list and then
    builds a directed graph (nx.DiGraph) where each edge has an attribute 'weight'.
    
    Parameters:
      pt_pdf (DataFrame): Must contain "line_id", "pt_id", and "prj_length".
      line_gdf (GeoDataFrame): Must contain "line_id", "geometry", and "encoding".
    
    Returns:
      A networkX DiGraph with edges added.
    """
    if is_direct:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    #将合并后的数组转换为带权重的边列表, 批量插入边和权重
    G.add_weighted_edges_from(edges)
    return G