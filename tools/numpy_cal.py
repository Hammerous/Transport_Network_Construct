import numpy as np

def find_nearest_intersect(point_arr: np.ndarray, line_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For each point in `point_arr`, compute its orthogonal projection onto every line segment in `line_arr`
    and select the projection with the minimum Euclidean distance.
    
    Parameters:
        point_arr (np.ndarray):
            Array of shape (n, 2) with point coordinates.
        line_arr (np.ndarray):
            Array of shape (m, 2, 2) where each row represents a line segment by its two endpoints.
    
    Returns:
        min_indices: np.ndarray of shape (n_points,)
            The index of the line (in line_arr) that is nearest to each point.
        attributes (np.ndarray):
            Array of shape (n, 4) where:
                - Column 0 is the projection length (a value in [0, 1]),
                - Column 1 is the Euclidean distance from the point to its projection.
                - Column 2 and 3 containing the X/Y coordinates of the projected points.
    """
    # Compute direction vectors for each line segment.
    line_vecs = line_arr[:, 1, :] - line_arr[:, 0, :]  # shape: (m, 2)
    
    # Compute the vector from each line's starting point to every point.
    diff = point_arr[np.newaxis, :, :] - line_arr[:, np.newaxis, 0, :]  # shape: (m, n, 2)
    
    # Compute projection lengths (t parameters) for each point-line pair.
    proj_lengths_all = np.einsum('ijk,ik->ij', diff, line_vecs) / np.einsum('ij,ij->i', line_vecs, line_vecs)[:, np.newaxis]
    proj_lengths_all = np.clip(proj_lengths_all, 0, 1)
    
    # Compute the projected points for each line segment.
    proj_points_all = line_arr[:, np.newaxis, 0, :] + proj_lengths_all[:, :, None] * line_vecs[:, np.newaxis, :]
    
    # Compute squared Euclidean distances between points and their projections.
    distances_all = np.sum((point_arr[np.newaxis, :, :] - proj_points_all)**2, axis=-1)
    
    # For each point, find the line segment with the minimum distance.
    min_indices = np.argmin(distances_all, axis=0)
    point_indices = np.arange(point_arr.shape[0])
    
    # Retrieve the selected projection lengths, distances, and projection points.
    proj_lengths = proj_lengths_all[min_indices, point_indices]
    distances = np.sqrt(distances_all[min_indices, point_indices])
    proj_points = proj_points_all[min_indices, point_indices]
    
    # Stack attributes into an (n, 4) array.
    attributes = np.column_stack((proj_lengths, distances, proj_points))
    return min_indices, attributes

def collect_arr(lst: list):
    return np.concatenate(lst)

# Example usage:
if __name__ == '__main__':
    # 10 points and 5 line segments for demonstration:
    points = np.random.rand(10, 2)
    lines = np.random.rand(5, 2, 2)
    nearest_line_indices, proj_lengths, distances, prj_points = find_nearest_intersect(points, lines)
    print("Nearest line indices:", nearest_line_indices)
    print("Projection lengths:", proj_lengths)
    print("Distances:", distances)
