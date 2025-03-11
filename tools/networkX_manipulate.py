import networkx as nx

def create_network_graph(edges: list, is_direct: bool, filename: str):
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
    print("Saving Edgelist ...")
    nx.write_weighted_edgelist(G, filename + ".edgelist")
    print("Edgelist Saved !!!")
    #return G