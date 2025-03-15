import tools.networkX_manipulate as ntx
import os, json, time

edgelist_file = 'topo_walk_lines.edgelist'
od_file = 'OD_file.csv'
has_direction = False
# ntx.numeric_val = 4
cut_off = 8000
self_dist = 199.47
cpus = 8

def timestamp():
    """Returns the current time formatted as a timestamp."""
    return time.strftime("[%Y-%m-%d %H:%M:%S]")

if __name__ == '__main__':
    print(f"{timestamp()}Loading Files ...")
    network = ntx.open_edgelist(edgelist_file, has_direction)
    od_df = ntx.pd_od_readin(od_file)

    o_set = set(od_df['O'].values)
    d_set = set(od_df['D'].values)

    print(f"{timestamp()}Querying ...")
    result_dict = ntx.source_shortest_path_length(G=network, origs=o_set, dests=d_set,
                                                weight='weight', cutoff_threshold=cut_off,
                                                self_weight=self_dist, cpus=cpus)
    
    print(f"{timestamp()}Saving File ...")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),\
        f'{os.path.splitext(os.path.basename(od_file))[0]}-{cut_off}.json'), 'w') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)