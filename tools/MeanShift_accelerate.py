from sklearn.cluster import MeanShift
import numpy as np

def point_cluster(bandwidth, num_in_block, point_coordinates, if_bin_seeding, if_cluster_all):
    # Convert GeoSeries to NumPy array efficiently
    point_coordinates = np.column_stack((point_coordinates.x.values, point_coordinates.y.values))
    if bandwidth and num_in_block:
        print('\rClustering Point Sequence ...  ', end='')
        ms = MeanShift(bandwidth=bandwidth, min_bin_freq=num_in_block,\
                        bin_seeding=if_bin_seeding, cluster_all=if_cluster_all, n_jobs=-1).fit(point_coordinates)
        labels = ms.labels_
        coords_clusters = {label: np.where(labels == label)[0] for label in np.unique(labels)}
        del ms
    else:
        coords_clusters = {-1: np.arange(point_coordinates.shape[0])}
    return coords_clusters