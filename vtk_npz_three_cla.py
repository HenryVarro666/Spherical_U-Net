import pyvista as pv
import numpy as np
import os
import re
from collections import deque
from tqdm import tqdm

def read_vtk(in_file):
    """
    读取 .vtk POLYDATA 文件

    参数:
        in_file (str): 文件名
    返回:
        dict: 包含 'vertices', 'faces', 和其他点数据的字典
    """
    try:
        polydata = pv.read(in_file)
    except Exception as e:
        print(f"读取 {in_file} 时出错: {e}")
        return None

    vertices = np.array(polydata.points)
    faces = np.array(polydata.faces).reshape(-1, 4)[:, 1:]

    data = {'vertices': vertices, 'faces': faces}

    # 提取线数据
    lines = polydata.lines
    if lines.size > 0:
        line_indices = lines.reshape((-1, 3))[:, 1:]  # Assuming lines are stored with a count + 2 indices
        data['line_indices'] = line_indices

    point_data = polydata.point_data
    for key, value in point_data.items():
        data[key] = np.array(value)

    return data

def build_adjacency_list(faces, num_vertices):
    """
    构建邻接列表以表示三角形网格的拓扑结构

    参数:
        faces (ndarray): 面的数组
        num_vertices (int): 顶点数量
    返回:
        list: 邻接列表
    """
    adj_list = [[] for _ in range(num_vertices)]
    for face in faces:
        for i in range(3):
            for j in range(i + 1, 3):
                adj_list[face[i]].append(face[j])
                adj_list[face[j]].append(face[i])
    return adj_list

def bfs_expand(line_indices, adjacency_list, width_steps):
    """
    使用广度优先搜索扩展线的邻域

    参数:
        line_indices (ndarray): 线的顶点索引
        adjacency_list (list): 邻接列表
        width_steps (int): 邻域宽度
    返回:
        set: 扩展后的顶点索引集合
    """
    visited = set()
    queue = deque()
    for idx in line_indices:
        queue.append((idx, 0))
        visited.add(idx)
    
    expanded_points = set(line_indices)
    
    while queue:
        current, depth = queue.popleft()
        if depth < width_steps:
            for neighbor in adjacency_list[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    expanded_points.add(neighbor)
    
    return expanded_points

def compute_line_mask(surface_points, faces, line_indices, width_steps=1):
    """
    计算线的掩码

    参数:
        surface_points (ndarray): 表面顶点
        faces (ndarray): 面的数组
        line_indices (ndarray): 线的顶点索引
        width_steps (int): 邻域宽度
    返回:
        ndarray: 掩码数组
    """
    num_vertices = len(surface_points)
    adjacency_list = build_adjacency_list(faces, num_vertices)
    expanded_points = bfs_expand(line_indices, adjacency_list, width_steps)
    
    mask = np.zeros(num_vertices, dtype=np.int32)
    mask[list(expanded_points)] = 1
    
    return mask

def one_hot_encode(labels, num_classes):
    """
    将标签转换为one-hot编码

    参数:
        labels (ndarray): 标签数组
        num_classes (int): 类别数
    返回:
        ndarray: one-hot编码的数组
    """
    return np.eye(num_classes)[labels]

def save_line_mask_npy(feature_file, sphe_file, line_file, output_dir):
    """
    保存线掩码为 .npz 文件

    参数:
        feature_file (str): 特征文件名
        sphe_file (str): 球面文件名
        line_file (str): 线文件名
        output_dir (str): 输出目录
    """
    # 读取球面数据和线数据
    sphere_data = read_vtk(sphe_file)
    line_data = read_vtk(line_file)

    if sphere_data is None or line_data is None:
        print(f"跳过 {feature_file} 因为球面数据或线数据读取失败。")
        return

    surface_points = sphere_data['vertices']
    faces = sphere_data['faces']
    line_indices_raw = line_data.get('line_indices', None)

    line_indices = line_indices_raw.flatten()

    # 计算线掩码
    line_mask = compute_line_mask(surface_points, faces, line_indices, width_steps=1)
    
    # 读取原始表面特征文件
    original_surface = pv.read(feature_file)

    # 提取顶点和面数据
    original_vertices = np.array(original_surface.points)
    original_faces = np.array(original_surface.faces)

    # 确保 faces 数组是正确的大小和格式
    faces = original_faces.reshape((-1, 4))
    if not np.all(faces[:, 0] == 3):
        raise ValueError("All faces should be triangles and have the first element as 3.")
    faces = faces.flatten()

    # 提取曲率和其他点数据
    original_curv = original_surface.point_data.get('curv', np.zeros(original_vertices.shape[0]))
    original_sulc = original_surface.point_data.get('sulc', np.zeros(original_vertices.shape[0]))

    # 创建单独的标签数组
    line_original_mask = np.zeros(original_vertices.shape[0], dtype=np.int32)
    line_original_mask[line_indices] = 2

    non_line_data = (line_original_mask == 0).astype(np.int32)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # one-hot编码
    combined_mask = np.maximum(line_original_mask, line_mask)
    one_hot_labels = one_hot_encode(combined_mask, 3)

    # 保存为 .npz 文件
    output_filename = os.path.join(output_dir, os.path.basename(feature_file).replace('.vtk', '_linemask_skeleton.npz'))

    np.savez(output_filename, 
             vertices=original_vertices, 
             faces=faces, 
             curv=original_curv, 
             sulc=original_sulc, 
             line_original_mask=one_hot_labels,
             line_mask=line_mask,
             non_line_data=non_line_data)

    print(f'Saving {output_filename}')

# 示例用法
if __name__ == "__main__":
    home_dir = '/media/lab/ef1e5021-01ef-4f9e-9cf7-950095b49199/HCP_fromFenqiang/'
    subjects = [subject for subject in os.listdir(home_dir) if re.match(r'^\d+$', subject)]
    
    for subject in tqdm(subjects, desc="Processing subjects"):
        for hemi in ['lh', 'rh']:
            feature_file = os.path.join(home_dir, subject, subject+'_recon_40962','surf', f'{subject}.{hemi}.InnerSurf.RegByFS.Resp40962.vtk')
            sphe_file = os.path.join(home_dir, subject, subject+'_recon_40962','surf', f'{subject}.{hemi}.SpheSurf.RegByFS.Resp40962.vtk')
            line_file = os.path.join(home_dir, subject, subject+'_gyralnet_island_40962', f'{hemi}_sphere_skelenton_allpoints_final.vtk')
            output_dir = '/home/lab/Documents/Spherical_U-Net/Test3'
            try:
                save_line_mask_npy(feature_file, sphe_file, line_file, output_dir)
            except Exception as e:
                print(f"处理 {subject} 时出错: {e}. 跳过此主体。")
