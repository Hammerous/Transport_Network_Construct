from sklearn.cluster import MeanShift
import numpy as np

def point_cluster(bandwidth: float, num_in_block: int, coords_array: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Clusters a set of points using the MeanShift algorithm with the specified bandwidth and number of points per block.

    This function first performs clustering on the entire set of points. If there are any orphan points (label -1),
    it re-clusters them with `cluster_all=True` to ensure all points are assigned to a cluster.

    Parameters:
    - bandwidth: The bandwidth parameter for the MeanShift algorithm, which controls the size of the clusters.
    - num_in_block: The minimum number of points required in each block for clustering.
    - coords_array: A 2D numpy array of coordinates (shape: [num_points, 2]) representing the points to be clustered.

    Returns:
    - A tuple containing:
      - `coords_clusters`: A dictionary where the keys are cluster labels and the values are arrays of indices of points in each cluster.
      - `orphans_clusters`: A dictionary of orphan clusters, where the keys are the new labels and the values are arrays of indices of re-clustered orphan points.
    """
    
    if not (bandwidth and num_in_block):  # If clustering is disabled
        return {}, {i: np.array([i]) for i in range(coords_array.shape[0])}
    
    print(f'Clustering Point Sequence ({coords_array.shape[0]} in sum)...')
    
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
        print(f'Clustering Orphan Points ({coords_clusters[-1].shape[0]} in sum)...')
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
        
    return coords_clusters, orphans_clusters