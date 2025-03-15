import tools.networkX_manipulate as ntx
import time, os

edgelist_file = 'Hybrid_Network.edgelist'
od_file = 'OD_file.csv'
has_direction = True
cpus = 2

def timestamp():
    """Returns the current time formatted as a timestamp."""
    return time.strftime("[%Y-%m-%d %H:%M:%S]")

if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),\
        f'{os.path.splitext(os.path.basename(od_file))[0]}_in_{os.path.splitext(os.path.basename(edgelist_file))[0]}.csv')
    print(f"{timestamp()}Loading Files ...")
    od_df = ntx.pd_od_readin(od_file)
    if od_df[['O', 'D']].isna().any().any():
        raise ValueError("Input OD pairs contains missing values.")
    network = ntx.open_edgelist(edgelist_file, has_direction)

    print(f"{timestamp()}Querying ...")
    query = ntx.shortest_path(network, od_df, weight='weight', cpus=cpus)
    
    print(f"{timestamp()}Saving File ...")
    ### under the same folder ensured
    ntx.save_paths(csv_path, paths=query, od_df=od_df,index=False)