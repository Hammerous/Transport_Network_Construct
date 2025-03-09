import numpy as np

def find_projection(pt_arr, line_vectors, line_vecs, nearby_lines_index):
    # 计算点到线段起点的向量
    #point_vecs = points_arr[np.newaxis, :, :] - line_vectors[:, np.newaxis, 0, :], line_vecs)
    # 计算投影长度(相对于原始长度的比例)
    proj_length = np.einsum('ijk,ik->ij', pt_arr[np.newaxis, :, :] - line_vectors[:, np.newaxis, 0, :], line_vecs) /       \
                   np.einsum('ij,ij->i', line_vecs, line_vecs)[:, np.newaxis]

    # 计算投影点
    proj_points = line_vectors[:, np.newaxis, 0, :] + proj_length[:, :, np.newaxis] * line_vecs[:, np.newaxis, :]
    del line_vecs

    # 创建条件掩码
    mask_below_zero = proj_length[:, :, np.newaxis] < 0
    mask_above_one = proj_length[:, :, np.newaxis] > 1

    # 更新投影点
    proj_points = np.where(mask_below_zero, line_vectors[:, np.newaxis, 0, :], proj_points)
    proj_points = np.where(mask_above_one, line_vectors[:, np.newaxis, 1, :], proj_points)    ### (line_idx, pt_idx, 2)
    del mask_below_zero, mask_above_one

    # 计算点到投影点的距离
    distances = np.linalg.norm(pt_arr[np.newaxis, :, :] - proj_points, axis=2)                      ### pt_idx dimensions, line_idx in element
    del proj_points
    
    line_idx_arr, proj_idx_arr = np.unique(np.argmin(distances, axis=0), return_inverse=True)
    nearby_lines_index = nearby_lines_index[line_idx_arr][proj_idx_arr]
    
    # 离散索引
    col_idx = np.arange(proj_idx_arr.shape[0])
    proj_length = proj_length[line_idx_arr, :][proj_idx_arr, col_idx]
    distances = distances[line_idx_arr, :][proj_idx_arr, col_idx]
    return nearby_lines_index, proj_length, distances