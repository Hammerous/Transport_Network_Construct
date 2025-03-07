import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

# Define the second function for loading lines
def loading_lines(lines, result):
    print('\rLoading Lines ...              ', end='')
    node_dict = {}
    count = 0
    def get_or_create_encoding(coord):
        nonlocal count
        if coord not in node_dict:
            node_dict[coord] = f'L{count}'
            count += 1
        return node_dict[coord]

    encoded_line_vectors = []
    array_lines = []
    for line in lines.geometry:
        for start, end in zip(line.coords[:-1], line.coords[1:]):
            encoded_line_vectors.append((get_or_create_encoding(start), get_or_create_encoding(end)))
            array_lines.append((start, end))

    lines = gpd.GeoDataFrame(geometry=[LineString(coords) for coords in array_lines], crs=lines.crs)
    # 将数据转换为numpy数组
    array_lines = np.array(array_lines, dtype=np.float64)
    encoded_line_vectors = np.array(encoded_line_vectors)

    # 根据start和end的坐标大小关系重新排列
    sorted_indices = np.lexsort((array_lines[:, :, 1], array_lines[:, :, 0]))
    array_lines = array_lines[np.arange(array_lines.shape[0])[:, None], sorted_indices]
    encoded_line_vectors = encoded_line_vectors[np.arange(encoded_line_vectors.shape[0])[:, None], sorted_indices]

    ### array_lines已经按照从左下到右上的顺序排列，encoded_line_vectors为对应顺序的线段端点
    all_line_vecs = array_lines[:, 1, :] - array_lines[:, 0, :]   # 计算线段的向量
    all_line_length = np.linalg.norm(all_line_vecs, axis=1)

    #返回数据
    result['lines'] = lines
    result['array_lines'] = array_lines
    result['encoded_line_vectors'] = encoded_line_vectors
    result['all_line_length'] = all_line_length
    result['all_line_vecs'] = all_line_vecs
    result['count'] = count