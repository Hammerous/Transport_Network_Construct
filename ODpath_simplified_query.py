import tools.networkX_manipulate as ntx
import time, os, json

edgelist_file = 'Hybrid_Network_simplified.edgelist'
merge_edges = 'Hybrid_Network_simplified.json'
od_file = 'OD_file.csv'
has_direction = True
cpus = 8

def timestamp():
    """Returns the current time formatted as a timestamp."""
    return time.strftime("[%Y-%m-%d %H:%M:%S]")

if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),\
        f'{os.path.splitext(os.path.basename(od_file))[0]}_in_{os.path.splitext(os.path.basename(edgelist_file))[0]}.csv')
    
    print(f"{timestamp()}Loading Files ...")
    with open(merge_edges, 'r', encoding='utf-8') as file:
        merge_edges = json.load(file)
    
    od_df = ntx.pd_od_readin(od_file)
    if od_df[['O', 'D']].isna().any().any():
        raise ValueError("Input OD pairs contains missing values.")
    network = ntx.open_edgelist(edgelist_file, has_direction)
    print("Number of edges loaded:", network.number_of_edges())

    print(f"{timestamp()}Querying ...")
    query = ntx.shortest_path(network, od_df, weight='weight', cpus=cpus)
    
    print(f"{timestamp()}Merged edges recovering ...")
    # query_extended = []
    # for dist, path in query:
    #     if len(path) > 1:
    #         path_node = ""
    #         for i in range(len(path) - 1):
    #             path_node += merge_edges.get(f"{path[i]}{path[i+1]}",f"{path[i]},{path[i+1]}").split(',')
    #         query_extended.append((dist, list(dict.fromkeys(path_node))))
    #     else:
    #         query_extended.append((dist, path))
    query_extended = [
    (dist,
     list(dict.fromkeys(
         [node
          for i in range(len(path) - 1)
          for node in merge_edges.get(f"{path[i]}{path[i+1]}", f"{path[i]},{path[i+1]}").split(',')
         ]
     ))
    ) if len(path) > 1 else (dist, path)
    for dist, path in query
    ]
    
    print(f"{timestamp()}Saving File ...")
    ### under the same folder ensured
    ntx.save_paths(csv_path, paths=query_extended, od_df=od_df,index=False)