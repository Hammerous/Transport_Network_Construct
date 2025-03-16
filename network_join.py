import tools.networkX_manipulate as ntx

joined_name = 'Hybrid_Network.edgelist'

walk_net = 'topo_walk_lines.edgelist'
walk_speed = 4 * 1e3/3600

sub_net = 'topo_subway_lines.edgelist'
sub_net_trans = 'subway_stop_trans.edgelist'
sub_stops = 'subway_stops.csv'
sub_wait_time = 5 * 60
sub_speed = 40 * 1e3/3600

bus_net = 'topo_bus_lines_fixed.edgelist'
bus_stops = 'bus_stops_fixed.csv'
bus_wait_time = 5 * 60
bus_speed = 30 * 1e3/3600

if __name__ == '__main__':
    print("Loading Edgelist Files ...")
    walk_net = ntx.pd_edgelist('topo_walk_lines.edgelist')
    sub_net = ntx.pd_edgelist('topo_subway_lines.edgelist')
    sub_net_trans = ntx.pd_edgelist('subway_stop_trans.edgelist')
    bus_net = ntx.pd_edgelist('topo_bus_lines_fixed.edgelist')
    print("Edgelist Files Loaded !!!")

    sub_stops = ntx.pd.read_csv(sub_stops, index_col='Sub_Id')
    bus_stops = ntx.pd.read_csv(bus_stops, index_col='Bus_Id')
    print("Stops Files Loaded !!!")

    walk_net = ntx.pd_avg_weight(walk_net, walk_speed)
    sub_net = ntx.pd_avg_weight(sub_net, sub_speed)
    bus_net = ntx.pd_avg_weight(bus_net, bus_speed)
    print("Net Speed Averaged !!!")
    
    sub_net = ntx.pd_intersect_reassign(sub_net, sub_stops.index, 'src', sub_wait_time)
    sub_net = ntx.pd_intersect_reassign(sub_net, sub_stops.index, 'end', sub_wait_time)
    bus_net = ntx.pd_intersect_reassign(bus_net, bus_stops.index, 'src', bus_wait_time)
    print("Transfer Time Assigned !!!")

    print("Merging Networks ...")
    walk_net = ntx.pd.concat([walk_net, bus_net, sub_net, sub_net_trans], ignore_index=True)

    print("Saving Edgelist ...")
    ntx.save_edgelist(joined_name, walk_net)


