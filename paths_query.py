import tools.networkX_manipulate as ntx
import os, json, time

edgelist_file = 'Network_Construct\Hybrid_Network_simplified.edgelist'
od_file = 'Network_Construct\_OD_file.csv'
has_direction = True
ntx.numeric_val = 1
cut_off = 60 * 60
self_dist = 180
cpus = 8

def timestamp():
    """Returns the current time formatted as a timestamp."""
    return time.strftime("[%Y-%m-%d %H:%M:%S]")

if __name__ == '__main__':
    print(f"{timestamp()}Loading Files ...")
    od_df = ntx.pd_od_readin(od_file)
    network = ntx.open_edgelist(edgelist_file, has_direction)
    print("Number of edges loaded:", network.number_of_edges())

    o_set = set(od_df['O'].values)
    d_set = set(od_df['D'].values)

    print(f"{timestamp()}Querying ...")
    result_dict = ntx.source_shortest_path_length(G=network, origs=o_set, dests=d_set,
                                                weight='weight', cutoff_threshold=cut_off,
                                                self_weight=self_dist, cpus=cpus)
    
    print(f"{timestamp()}Saving File ...")
    ### under the same folder ensured
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),\
        f'{os.path.splitext(os.path.basename(od_file))[0]}-{cut_off}.json'), 'w') as f:
        json.dump(result_dict, f, ensure_ascii=False)