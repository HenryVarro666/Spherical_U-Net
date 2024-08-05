# import pyvista as pv
# import numpy as np
# import os
# import re
# from collections import deque
# from tqdm import tqdm

# def read_vtk(in_file):
#     """
#     读取 .vtk POLYDATA 文件

#     参数:
#         in_file (str): 文件名
#     返回:
#         dict: 包含 'vertices', 'faces', 和其他点数据的字典
#     """
#     try:
#         polydata = pv.read(in_file)
#     except Exception as e:
#         print(f"读取 {in_file} 时出错: {e}")
#         return None

#     vertices = np.array(polydata.points)
#     faces = np.array(polydata.faces).reshape(-1, 4)[:, 1:]

#     data = {'vertices': vertices, 'faces': faces}

#     # 提取线数据
#     lines = polydata.lines
#     if lines.size > 0:
#         line_indices = lines.reshape((-1, 3))[:, 1:]  # Assuming lines are stored with a count + 2 indices
#         data['line_indices'] = line_indices

#     point_data = polydata.point_data
#     for key, value in point_data.items():
#         data[key] = np.array(value)

#     return data

# def build_adjacency_list(faces, num_vertices):
#     """
#     构建邻接列表以表示三角形网格的拓扑结构

#     参数:
#         faces (ndarray): 面的数组
#         num_vertices (int): 顶点数量
#     返回:
#         list: 邻接列表
#     """
#     adj_list = [[] for _ in range(num_vertices)]
#     for face in faces:
#         for i in range(3):
#             for j in range(i + 1, 3):
#                 adj_list[face[i]].append(face[j])
#                 adj_list[face[j]].append(face[i])
#     return adj_list

# def compute_ring_numbers(line_indices, adjacency_list):
#     """
#     计算每个点到线数据的最小环号

#     参数:
#         line_indices (ndarray): 线的顶点索引
#         adjacency_list (list): 邻接列表
#     返回:
#         ndarray: 环号数组
#     """
#     num_vertices = len(adjacency_list)
#     ring_numbers = np.full(num_vertices, np.inf, dtype=np.float32)
#     line_indices = line_indices.flatten()
#     ring_numbers[line_indices] = 0

#     queue = deque(line_indices)
#     while queue:
#         current = queue.popleft()
#         current_ring = ring_numbers[current]
#         for neighbor in adjacency_list[current]:
#             if ring_numbers[neighbor] == np.inf:
#                 ring_numbers[neighbor] = current_ring + 1
#                 queue.append(neighbor)

#     return ring_numbers

# def compute_line_mask(surface_points, faces, line_indices):
#     """
#     计算线的掩码和n-ring距离

#     参数:
#         surface_points (ndarray): 表面顶点
#         faces (ndarray): 面的数组
#         line_indices (ndarray): 线的顶点索引
#     返回:
#         ndarray: 掩码数组
#     """
#     num_vertices = len(surface_points)
#     mask = np.zeros(num_vertices, dtype=np.int32)
#     mask[line_indices] = 2

#     adjacency_list = build_adjacency_list(faces, num_vertices)
#     ring_numbers = compute_ring_numbers(line_indices, adjacency_list)
#     mask[ring_numbers == 1] = 1  # 设置距离为1环的点

#     return mask, ring_numbers

# def save_line_mask_npz(feature_file, sphe_file, line_file, output_dir):
#     """
#     保存线掩码和n-ring距离为 .npz 文件

#     参数:
#         feature_file (str): 特征文件名
#         sphe_file (str): 球面文件名
#         line_file (str): 线文件名
#         output_dir (str): 输出目录
#     """
#     # 读取球面数据和线数据
#     sphere_data = read_vtk(sphe_file)
#     line_data = read_vtk(line_file)

#     if sphere_data is None or line_data is None:
#         print(f"跳过 {feature_file} 因为球面数据或线数据读取失败。")
#         return

#     surface_points = sphere_data['vertices']
#     faces = sphere_data['faces']
#     line_indices_raw = line_data.get('line_indices', None)

#     line_indices = line_indices_raw.flatten()

#     # 计算线掩码和n-ring距离
#     line_mask, ring_distances = compute_line_mask(surface_points, faces, line_indices)

#     # 确保线数据点的ring_distances为0
#     ring_distances[line_indices] = 0

#     # 读取原始表面特征文件
#     original_surface = pv.read(feature_file)

#     # 提取顶点和面数据
#     original_vertices = np.array(original_surface.points)
#     original_faces = np.array(original_surface.faces)

#     # 确保 faces 数组是正确的大小和格式
#     faces = original_faces.reshape((-1, 4))
#     if not np.all(faces[:, 0] == 3):
#         print(f"处理 {feature_file} 时出错: All faces should be triangles and have the first element as 3.")
#         return
#     faces = faces.flatten()

#     # 提取曲率和其他点数据
#     original_curv = original_surface.point_data.get('curv', np.zeros(original_vertices.shape[0]))
#     original_sulc = original_surface.point_data.get('sulc', np.zeros(original_vertices.shape[0]))

#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)

#     # 创建 PyVista 网格
#     polydata = pv.PolyData(original_vertices, faces)
#     polydata.point_data['curv'] = original_curv
#     polydata.point_data['sulc'] = original_sulc
#     polydata.point_data['line_mask'] = line_mask
#     polydata.point_data['ring_distances'] = ring_distances

#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)

#     # 保存为 .vtk 文件
#     output_filename = os.path.join(output_dir, os.path.basename(feature_file).replace('.vtk', '_linemask_skeleton.vtk'))

#     polydata.save(output_filename)

#     print(f'Saving {output_filename}')

# # 示例用法
# if __name__ == "__main__":
#     home_dir = '/media/lab/ef1e5021-01ef-4f9e-9cf7-950095b49199/HCP_fromFenqiang/'
#     subjects = [subject for subject in os.listdir(home_dir) if re.match(r'^\d+$', subject)]

#     for subject in tqdm(subjects, desc="Processing subjects"):
#         for hemi in ['lh', 'rh']:
#             feature_file = os.path.join(home_dir, subject, subject+'_recon_40962','surf', f'{subject}.{hemi}.InnerSurf.RegByFS.Resp40962.vtk')
#             sphe_file = os.path.join(home_dir, subject, subject+'_recon_40962','surf', f'{subject}.{hemi}.SpheSurf.RegByFS.Resp40962.vtk')
#             line_file = os.path.join(home_dir, subject, subject+'_gyralnet_island_40962', f'{hemi}_sphere_skelenton_allpoints_final.vtk')
#             output_dir = '/home/lab/Documents/Spherical_U-Net/Test_cla_reg'
#             try:
#                 save_line_mask_npz(feature_file, sphe_file, line_file, output_dir)
#             except Exception as e:
#                 print(f"处理 {subject} 时出错: {e}. 跳过此主体。")

##########################################################################################

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

def compute_ring_numbers(line_indices, adjacency_list):
    """
    计算每个点到线数据的最小环号

    参数:
        line_indices (ndarray): 线的顶点索引
        adjacency_list (list): 邻接列表
    返回:
        ndarray: 环号数组
    """
    num_vertices = len(adjacency_list)
    ring_numbers = np.full(num_vertices, np.inf, dtype=np.float32)
    line_indices = line_indices.flatten()

    ring_numbers[line_indices] = 0

    queue = deque(line_indices)
    while queue:
        current = queue.popleft()
        current_ring = ring_numbers[current]
        for neighbor in adjacency_list[current]:
            if ring_numbers[neighbor] == np.inf:
                ring_numbers[neighbor] = current_ring + 1
                queue.append(neighbor)

    return ring_numbers

def compute_line_mask(surface_points, faces, line_indices):
    """
    计算线的掩码和n-ring距离

    参数:
        surface_points (ndarray): 表面顶点
        faces (ndarray): 面的数组
        line_indices (ndarray): 线的顶点索引
    返回:
        ndarray: 掩码数组
    """
    num_vertices = len(surface_points)
    mask = np.zeros(num_vertices, dtype=np.int32)
    mask[line_indices] = 2

    adjacency_list = build_adjacency_list(faces, num_vertices)
    ring_numbers = compute_ring_numbers(line_indices, adjacency_list)
    mask[ring_numbers == 1] = 1  # 设置距离为1环的点

    return mask, ring_numbers

def save_line_mask_npz(feature_file, sphe_file, line_file, output_dir):
    """
    保存线掩码和n-ring距离为 .npz 文件

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

    # 计算线掩码和n-ring距离
    line_mask, ring_distances = compute_line_mask(surface_points, faces, line_indices)

    # 确保线数据点的ring_distances为0
    ring_distances[line_indices] = 0

    # 读取原始表面特征文件
    original_surface = pv.read(feature_file)

    # 提取顶点和面数据
    original_vertices = np.array(original_surface.points)
    original_faces = np.array(original_surface.faces)

    # 确保 faces 数组是正确的大小和格式
    faces = original_faces.reshape((-1, 4))
    if not np.all(faces[:, 0] == 3):
        print(f"处理 {feature_file} 时出错: All faces should be triangles and have the first element as 3.")
        return
    faces = faces.flatten()

    # 提取曲率和其他点数据
    original_curv = original_surface.point_data.get('curv', np.zeros(original_vertices.shape[0]))
    original_sulc = original_surface.point_data.get('sulc', np.zeros(original_vertices.shape[0]))

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存为 .npz 文件
    output_filename = os.path.join(output_dir, os.path.basename(feature_file).replace('.vtk', '_linemask_skeleton.npz'))

    np.savez(output_filename,
             vertices=original_vertices,
             faces=faces,
             curv=original_curv,
             sulc=original_sulc,
             line_mask=line_mask,
             ring_distances=ring_distances)

    print(f'Saving {output_filename}')

# 示例用法
if __name__ == "__main__":
    home_dir = '/media/lab/ef1e5021-01ef-4f9e-9cf7-950095b49199/HCP_fromFenqiang/'
    subjects = [subject for subject in os.listdir(home_dir) if re.match(r'^\d+$', subject)]

    for subject in tqdm(subjects, desc="Processing subjects"):
        # for hemi in ['lh', 'rh']:
        for hemi in ['lh']:

            feature_file = os.path.join(home_dir, subject, subject+'_recon_40962','surf', f'{subject}.{hemi}.InnerSurf.RegByFS.Resp40962.vtk')
            sphe_file = os.path.join(home_dir, subject, subject+'_recon_40962','surf', f'{subject}.{hemi}.SpheSurf.RegByFS.Resp40962.vtk')
            line_file = os.path.join(home_dir, subject, subject+'_gyralnet_island_40962', f'{hemi}_sphere_skelenton_allpoints_final.vtk')
            output_dir = '/home/lab/Documents/Spherical_U-Net/Test_cla_reg'
            try:
                save_line_mask_npz(feature_file, sphe_file, line_file, output_dir)
            except Exception as e:
                print(f"处理 {subject} 时出错: {e}. 跳过此主体。")
