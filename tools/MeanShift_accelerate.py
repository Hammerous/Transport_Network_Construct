from sklearn.cluster import MeanShift
import numpy as np

def point_cluster(bandwidth, num_in_block, point_coordinates):
    # Convert GeoSeries to a NumPy array efficiently
    coords_array = np.column_stack((point_coordinates.x.values, point_coordinates.y.values))
    
    if bandwidth and num_in_block:
        print('Clustering Point Sequence ({0} in sum)...  '.format(coords_array.shape[0]))
        # First clustering using non-cluster-all strategy.
        ms = MeanShift(bandwidth=bandwidth,
                            min_bin_freq=num_in_block,
                            bin_seeding=True,
                            cluster_all=False,
                            n_jobs=-1).fit(coords_array)
        labels = ms.labels_
        # Build clusters: keys are labels and values are indices of points in that cluster.
        coords_clusters = {label: np.where(labels == label)[0] for label in np.unique(labels)}
        orphans_clusters = {}
        
        # If there is an orphan cluster (-1), re-cluster those points with cluster_all=True.
        if -1 in coords_clusters:
            print('Clustering Orphan Points ({0} in sum)...  '.format(coords_clusters[-1].shape[0]))
            orphan_indices = coords_clusters[-1]
            orphan_coords = coords_array[orphan_indices]
            ms = MeanShift(bandwidth=bandwidth,
                                min_bin_freq=1,
                                bin_seeding=True,
                                cluster_all=True,
                                n_jobs=-1).fit(orphan_coords)
            orphan_labels = ms.labels_
            # Map the new orphan labels to the corresponding original indices.
            orphans_clusters = {label: orphan_indices[np.where(orphan_labels == label)[0]]
                                for label in np.unique(orphan_labels)}
            # Remove the orphan cluster from the main clusters.
            del coords_clusters[-1]            
    else:
        # In case clustering function is disabled,
        # treat all points as orphans.
        coords_clusters = {}
        orphans_clusters = {-1: np.arange(coords_array.shape[0])}
        
    return coords_clusters, orphans_clusters, coords_array