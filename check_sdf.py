# ## Path: check_sdf.py
# ## Check zero distance values

# import pyvista as pv
# import numpy as np
# from scipy.spatial import cKDTree
# import os

# def read_vtk(in_file):
#     """
#     Read .vtk POLYDATA file
    
#     in_file: string,  the filename
#     Out: dictionary, 'vertices', 'faces', 'sulc', 'curv', ...
#     """
#     polydata = pv.read(in_file)
#     vertices = np.array(polydata.points)
#     faces = np.array(polydata.faces).reshape(-1, 4)[:, 1:]
    
#     data = {'vertices': vertices, 'faces': faces}
    
#     point_data = polydata.point_data
#     for key, value in point_data.items():
#         data[key] = np.array(value)
    
#     return data

# def compute_sdf(surface_points, line_points):
#     tree = cKDTree(line_points)
#     distances, _ = tree.query(surface_points)
#     return distances

# def save_sdf_vtk(surf_file, line_file, output_file, log_file):
#     if not os.path.exists(surf_file) or not os.path.exists(line_file):
#         with open(log_file, 'a') as log:
#             log.write(f'Skipping: {surf_file} or {line_file} does not exist\n')
#         return

#     surface_data = read_vtk(surf_file)
#     line_data = read_vtk(line_file)

#     surface_points = surface_data['vertices']
#     line_points = line_data['vertices']

#     sdf_values = compute_sdf(surface_points, line_points)

#     # 检查零距离值
#     zero_distance_indices = np.where(sdf_values == 0)[0]
#     if len(zero_distance_indices) > 0:
#         print(f"Points with zero distance: {surface_points[zero_distance_indices]}")
    
#     surf = pv.read(surf_file)
#     surf['sdf'] = sdf_values

#     surf.save(output_file, binary=False)

# if __name__ == '__main__':
#     root = '/mnt/d/Spherical_U-Net/vtk_files'
#     for subject in os.listdir(root):
#         if subject.isdigit():
#             for hemi in ['lh']:
#                 surf_file = os.path.join(root, subject, f'{subject}_recon_40962', 'surf', f'{subject}.{hemi}.InnerSurf.RegByFS.Resp40962.vtk')
#                 line_file = os.path.join(root, subject, f'{subject}_gyralnet_island_40962', f'{hemi}_surf_skelenton_allpoints_final.vtk')
#                 output_file = os.path.join(root, f'{subject}_{hemi}_sdf.vtk')
#                 log_file = os.path.join(root, 'sdf.log')
#                 save_sdf_vtk(surf_file, line_file, output_file, log_file)

#######################################################

# ## The sdf values are normalized and saved in the output file
# ## The distance is Euclidean distance

# import pyvista as pv
# import numpy as np
# from scipy.spatial import cKDTree
# import os

# def read_vtk(in_file):
#     """
#     Read .vtk POLYDATA file
    
#     in_file: string,  the filename
#     Out: dictionary, 'vertices', 'faces', 'sulc', 'curv', ...
#     """
#     polydata = pv.read(in_file)
#     vertices = np.array(polydata.points)
#     faces = np.array(polydata.faces).reshape(-1, 4)[:, 1:]
    
#     data = {'vertices': vertices, 'faces': faces}
    
#     point_data = polydata.point_data
#     for key, value in point_data.items():
#         data[key] = np.array(value)
    
#     return data

# def compute_sdf(surface_points, line_points):
#     tree = cKDTree(line_points)
#     distances, _ = tree.query(surface_points)
#     return distances

# def normalize_values(values):
#     min_val = np.min(values)
#     max_val = np.max(values)
#     if max_val == min_val:
#         return np.zeros_like(values)
#     normalized_values = (values - min_val) / (max_val - min_val)
#     return normalized_values

# def save_sdf_vtk(surf_file, line_file, output_file, log_file):
#     if not os.path.exists(surf_file) or not os.path.exists(line_file):
#         with open(log_file, 'a') as log:
#             log.write(f'Skipping: {surf_file} or {line_file} does not exist\n')
#         return

#     surface_data = read_vtk(surf_file)
#     line_data = read_vtk(line_file)

#     surface_points = surface_data['vertices']
#     line_points = line_data['vertices']


#     sdf_values = compute_sdf(surface_points, line_points)
#     print(f"SDF values: {sdf_values[:10]}")

#     surf = pv.read(surf_file)
#     surf['sdf'] = sdf_values

#     normalized_sdf_values = normalize_values(sdf_values)
#     # print(f"Normalized SDF values: {normalized_sdf_values[:10]}")  # 打印前10个归一化后的SDF值进行检查

#     surf['sdf_norm'] = normalized_sdf_values

#     surf.save(output_file, binary=False)

# if __name__ == '__main__':
#     root = '/mnt/d/Spherical_U-Net/vtk_files'
#     for subject in os.listdir(root):
#         if subject.isdigit():
#             for hemi in ['lh']:
#                 surf_file = os.path.join(root, subject, f'{subject}_recon_40962', 'surf', f'{subject}.{hemi}.InnerSurf.RegByFS.Resp40962.vtk')
#                 line_file = os.path.join(root, subject, f'{subject}_gyralnet_island_40962', f'{hemi}_surf_skelenton_allpoints_final.vtk')
#                 output_file = os.path.join(root, f'{subject}_{hemi}_sdf.vtk')
#                 log_file = os.path.join(root, 'sdf.log')
#                 save_sdf_vtk(surf_file, line_file, output_file, log_file)


#######################################################

# ## The distance is the shortest path distance on the surface mesh
# ## Ignore. Takes too long to compute

# import pyvista as pv
# import numpy as np
# import os
# import networkx as nx
# from scipy.spatial import cKDTree

# def read_vtk(in_file):
#     """
#     Read .vtk POLYDATA file
    
#     in_file: string,  the filename
#     Out: dictionary, 'vertices', 'faces', 'sulc', 'curv', ...
#     """
#     polydata = pv.read(in_file)
#     vertices = np.array(polydata.points)
#     faces = np.array(polydata.faces).reshape(-1, 4)[:, 1:]
    
#     data = {'vertices': vertices, 'faces': faces}
    
#     point_data = polydata.point_data
#     for key, value in point_data.items():
#         data[key] = np.array(value)
    
#     return data

# def compute_surface_distance(surface_points, line_points, faces):
#     G = nx.Graph()

#     for face in faces:
#         for i in range(len(face)):
#             for j in range(i + 1, len(face)):
#                 u, v = face[i], face[j]
#                 dist = np.linalg.norm(surface_points[u] - surface_points[v])
#                 G.add_edge(u, v, weight=dist)

#     tree = cKDTree(surface_points)
#     line_point_indices = tree.query(line_points, k=1)[1]

#     distances = []
#     for lp_idx in line_point_indices:
#         dists = []
#         for sp_idx in range(len(surface_points)):
#             try:
#                 d = nx.shortest_path_length(G, source=sp_idx, target=lp_idx, weight='weight')
#                 dists.append(d)
#             except nx.NetworkXNoPath:
#                 dists.append(np.inf)
#         distances.append(min(dists))
    
#     return np.array(distances)

# def save_sdf_vtk(surf_file, line_file, output_file, log_file):
#     if not os.path.exists(surf_file) or not os.path.exists(line_file):
#         with open(log_file, 'a') as log:
#             log.write(f'Skipping: {surf_file} or {line_file} does not exist\n')
#         return

#     surface_data = read_vtk(surf_file)
#     line_data = read_vtk(line_file)

#     surface_points = surface_data['vertices']
#     line_points = line_data['vertices']
#     faces = surface_data['faces']

#     sdf_values = compute_surface_distance(surface_points, line_points, faces)

#     surf = pv.read(surf_file)
#     surf['sdf'] = sdf_values

#     surf.save(output_file, binary=False)

# if __name__ == '__main__':
#     root = '/mnt/d/Spherical_U-Net/vtk_files'
#     for subject in os.listdir(root):
#         if subject.isdigit():
#             for hemi in ['lh']:
#                 surf_file = os.path.join(root, subject, f'{subject}_recon_40962', 'surf', f'{subject}.{hemi}.InnerSurf.RegByFS.Resp40962.vtk')
#                 line_file = os.path.join(root, subject, f'{subject}_gyralnet_island_40962', f'{hemi}_surf_skelenton_allpoints_final.vtk')
#                 output_file = os.path.join(root, f'{subject}_{hemi}_sdf.vtk')
#                 log_file = os.path.join(root, 'sdf.log')
#                 save_sdf_vtk(surf_file, line_file, output_file, log_file)

#######################################################

# ## joblib is used to parallelize the computation of geodesic distances
# ## Failed. Takes too long to compute

# import pyvista as pv
# import numpy as np
# from scipy.spatial import cKDTree
# import os
# import networkx as nx
# from joblib import Parallel, delayed

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

#     point_data = polydata.point_data
#     for key, value in point_data.items():
#         data[key] = np.array(value)

#     return data

# def build_surface_graph(vertices, faces):
#     """
#     构建表示表面网格的图

#     参数:
#         vertices (numpy.ndarray): 顶点坐标数组
#         faces (numpy.ndarray): 面数组
#     返回:
#         networkx.Graph: 表面网格图
#     """
#     G = nx.Graph()
#     for face in faces:
#         for i in range(len(face)):
#             for j in range(i + 1, len(face)):
#                 u = face[i]
#                 v = face[j]
#                 distance = np.linalg.norm(vertices[u] - vertices[v])
#                 G.add_edge(u, v, weight=distance)
#     return G

# def compute_geodesic_distance(surface_graph, point_index, nearest_line_point_index):
#     """
#     计算单个点到最近线点的曲面距离

#     参数:
#         surface_graph (networkx.Graph): 表面网格图
#         point_index (int): 表面点的索引
#         nearest_line_point_index (int): 最近线点的索引
#     返回:
#         float: 曲面距离
#     """
#     shortest_path_length = nx.single_source_dijkstra_path_length(surface_graph, point_index, weight='weight')
#     return shortest_path_length[nearest_line_point_index]

# def compute_geodesic_distances_parallel(surface_graph, surface_points, line_points, n_jobs=-1):
#     """
#     并行计算表面点到线点的曲面距离

#     参数:
#         surface_graph (networkx.Graph): 表面网格图
#         surface_points (numpy.ndarray): 表面点坐标数组
#         line_points (numpy.ndarray): 线点坐标数组
#         n_jobs (int): 并行任务数，默认为-1，表示使用所有可用的CPU核心
#     返回:
#         numpy.ndarray: 每个表面点到最近线点的曲面距离数组
#     """
#     tree = cKDTree(line_points)
#     _, nearest_indices = tree.query(surface_points)

#     geodesic_distances = Parallel(n_jobs=n_jobs)(delayed(compute_geodesic_distance)(surface_graph, i, nearest_indices[i]) for i in range(len(surface_points)))

#     return np.array(geodesic_distances)

# def normalize_values(values):
#     """
#     归一化距离值到 0 和 1 之间

#     参数:
#         values (numpy.ndarray): 距离值数组
#     返回:
#         numpy.ndarray: 归一化的距离值数组
#     """
#     min_val = np.min(values)
#     max_val = np.max(values)
#     if max_val == min_val:
#         print("所有值都相同。返回零值。")
#         return np.zeros_like(values)
#     normalized_values = (values - min_val) / (max_val - min_val)
#     return normalized_values

# def save_sdf_vtk(surf_file, line_file, output_file, log_file, n_jobs=-1):
#     """
#     读取表面和线数据，计算 SDF 值，归一化并保存到新 .vtk 文件中

#     参数:
#         surf_file (str): 表面文件路径
#         line_file (str): 线文件路径
#         output_file (str): 输出文件路径
#         log_file (str): 日志文件路径
#         n_jobs (int): 并行任务数，默认为-1，表示使用所有可用的CPU核心
#     """
#     if not os.path.exists(surf_file) or not os.path.exists(line_file):
#         with open(log_file, 'a') as log:
#             log.write(f'跳过：{surf_file} 或 {line_file} 不存在\n')
#         return

#     surface_data = read_vtk(surf_file)
#     if surface_data is None:
#         with open(log_file, 'a') as log:
#             log.write(f'读取表面文件 {surf_file} 时出错\n')
#         return

#     line_data = read_vtk(line_file)
#     if line_data is None:
#         with open(log_file, 'a') as log:
#             log.write(f'读取线文件 {line_file} 时出错\n')
#         return

#     surface_points = surface_data['vertices']
#     line_points = line_data['vertices']
#     faces = surface_data['faces']

#     surface_graph = build_surface_graph(surface_points, faces)
#     geodesic_distances = compute_geodesic_distances_parallel(surface_graph, surface_points, line_points, n_jobs=n_jobs)

#     print(f"曲面距离值: {geodesic_distances[:10]}")

#     surf = pv.read(surf_file)
#     surf['sdf'] = geodesic_distances

#     normalized_sdf_values = normalize_values(geodesic_distances)
#     print(f"归一化的曲面距离值: {normalized_sdf_values[:10]}")  # 检查归一化的曲面距离值

#     surf['sdf_norm'] = normalized_sdf_values

#     try:
#         surf.save(output_file, binary=False)
#     except Exception as e:
#         with open(log_file, 'a') as log:
#             log.write(f'保存文件 {output_file} 时出错: {e}\n')
#         return

# if __name__ == '__main__':
#     root = '/mnt/d/Spherical_U-Net/vtk_files'
#     for subject in os.listdir(root):
#         if subject.isdigit():
#             for hemi in ['lh']:
#                 surf_file = os.path.join(root, subject, f'{subject}_recon_40962', 'surf', f'{subject}.{hemi}.InnerSurf.RegByFS.Resp40962.vtk')
#                 line_file = os.path.join(root, subject, f'{subject}_gyralnet_island_40962', f'{hemi}_surf_skelenton_allpoints_final.vtk')
#                 output_file = os.path.join(root, f'{subject}_{hemi}_sdf.vtk')
#                 log_file = os.path.join(root, 'sdf.log')
#                 save_sdf_vtk(surf_file, line_file, output_file, log_file, n_jobs=14)


#######################################################

# ## The distance is the Euclidean distance
# ## The sdf values are normalized and saved in the output file
# ## Add debug print statements

# import pyvista as pv
# import numpy as np
# from scipy.spatial import cKDTree
# import os

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

#     # 提取line数据
#     lines = polydata.lines
#     if lines.size > 0:
#         line_indices = lines.reshape((-1, 3))[:, 1:]  # Assuming lines are stored with a count + 2 indices
#         data['line_indices'] = line_indices

#     return data

# def extract_line_points(line_indices, vertices):
#     """
#     从线索引中提取线点

#     参数:
#         line_indices (numpy.ndarray): 线索引数组
#         vertices (numpy.ndarray): 顶点坐标数组
#     返回:
#         numpy.ndarray: 线点的坐标数组
#     """
#     line_points = vertices[line_indices].reshape(-1, 3)
#     return line_points

# def compute_sdf(surface_points, line_points):
#     """
#     计算表面点到线点的最小距离

#     参数:
#         surface_points (numpy.ndarray): 表面点的坐标数组
#         line_points (numpy.ndarray): 线点的坐标数组
#     返回:
#         numpy.ndarray: 每个表面点到最近线点的距离数组
#     """
#     if len(line_points) == 0:
#         print("未提供线点。")
#         return np.full(len(surface_points), np.inf)

#     tree = cKDTree(line_points)
#     distances, _ = tree.query(surface_points)
#     print(f"计算的距离值: {distances[:10]}")  # 添加打印语句检查计算的距离值
#     return distances

# def normalize_values(values):
#     """
#     归一化距离值到 0 和 1 之间

#     参数:
#         values (numpy.ndarray): 距离值数组
#     返回:
#         numpy.ndarray: 归一化的距离值数组
#     """
#     min_val = np.min(values)
#     max_val = np.max(values)
#     print(f"距离值的最小值: {min_val}, 最大值: {max_val}")  # 添加打印语句检查最小值和最大值
#     if max_val == min_val:
#         print("所有值都相同。返回零值。")
#         return np.zeros_like(values)
#     normalized_values = (values - min_val) / (max_val - min_val)
#     return normalized_values

# def save_sdf_vtk(surf_file, sphere_file, line_file, output_file, log_file):
#     """
#     读取表面和线数据，计算 SDF 值，归一化并保存到新 .vtk 文件中

#     参数:
#         surf_file (str): 表面文件路径
#         line_file (str): 线文件路径
#         output_file (str): 输出文件路径
#         log_file (str): 日志文件路径
#     """
#     if not os.path.exists(surf_file) or not os.path.exists(line_file):
#         with open(log_file, 'a') as log:
#             log.write(f'跳过：{surf_file} 或 {line_file} 不存在\n')
#         return

#     surface_data = read_vtk(surf_file)
#     if surface_data is None:
#         with open(log_file, 'a') as log:
#             log.write(f'读取表面文件 {surf_file} 时出错\n')
#         return

#     line_data = read_vtk(line_file)
#     if line_data is None:
#         with open(log_file, 'a') as log:
#             log.write(f'读取线文件 {line_file} 时出错\n')
#         return

#     surface_points = surface_data['vertices']
#     line_indices = line_data.get('line_indices', None)
    
#     if line_indices is None:
#         with open(log_file, 'a') as log:
#             log.write(f'线文件 {line_file} 中缺少线索引数据\n')
#         return

#     line_points = extract_line_points(line_indices, surface_points)

#     # 打印表面点和线点的一些示例数据
#     print(f"表面点数: {len(surface_points)}, 线点数: {len(line_points)}")
#     print(f"表面点示例: {surface_points[:5]}")
#     print(f"线点示例: {line_points[:5]}")

#     if len(surface_points) == 0 or len(line_points) == 0:
#         with open(log_file, 'a') as log:
#             log.write(f'错误：表面点或线点数据为空\n')
#         return

#     sdf_values = compute_sdf(surface_points, line_points)
#     if np.all(sdf_values == np.inf):
#         with open(log_file, 'a') as log:
#             log.write(f'错误：未能为 {surf_file} 计算有效的 SDF 值\n')
#         return

#     print(f"SDF 值: {sdf_values[:10]}")

#     surf = pv.read(surf_file)
#     surf['sdf'] = sdf_values

#     normalized_sdf_values = normalize_values(sdf_values)
#     print(f"归一化的 SDF 值: {normalized_sdf_values[:10]}")  # 检查归一化的 SDF 值

#     surf['sdf_norm'] = normalized_sdf_values

#     try:
#         surf.save(output_file, binary=False)
#     except Exception as e:
#         with open(log_file, 'a') as log:
#             log.write(f'保存文件 {output_file} 时出错: {e}\n')
#         return

# if __name__ == '__main__':
#     root = '/mnt/d/Spherical_U-Net/vtk_files'
#     for subject in os.listdir(root):
#         if subject.isdigit():
#             for hemi in ['lh']:
#                 surf_file = os.path.join(root, subject, f'{subject}_recon_40962', 'surf', f'{subject}.{hemi}.SpheSurf.RegByFS.Resp40962.vtk')
#                 line_file = os.path.join(root, subject, f'{subject}_gyralnet_island_40962', f'{hemi}_sphere_skelenton_allpoints_final.vtk')
#                 output_file = os.path.join(root, f'{subject}_{hemi}_sdf.vtk')
#                 log_file = os.path.join(root, 'sdf.log')
#                 save_sdf_vtk(surf_file, line_file, output_file, log_file)



#######################################################

import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree
import os

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

    # 提取line数据
    lines = polydata.lines
    if lines.size > 0:
        line_indices = lines.reshape((-1, 3))[:, 1:]  # Assuming lines are stored with a count + 2 indices
        data['line_indices'] = line_indices

    return data

def extract_line_points(line_indices, vertices):
    """
    从线索引中提取线点

    参数:
        line_indices (numpy.ndarray): 线索引数组
        vertices (numpy.ndarray): 顶点坐标数组
    返回:
        numpy.ndarray: 线点的坐标数组
    """
    line_points = vertices[line_indices].reshape(-1, 3)
    return line_points

def compute_sdf(surface_points, line_points):
    """
    计算表面点到线点的最小距离

    参数:
        surface_points (numpy.ndarray): 表面点的坐标数组
        line_points (numpy.ndarray): 线点的坐标数组
    返回:
        numpy.ndarray: 每个表面点到最近线点的距离数组
    """
    if len(line_points) == 0:
        print("未提供线点。")
        return np.full(len(surface_points), np.inf)

    tree = cKDTree(line_points)
    distances, _ = tree.query(surface_points)
    print(f"计算的距离值: {distances[:10]}")  # 添加打印语句检查计算的距离值
    return distances

def normalize_values(values):
    """
    归一化距离值到 0 和 1 之间

    参数:
        values (numpy.ndarray): 距离值数组
    返回:
        numpy.ndarray: 归一化的距离值数组
    """
    min_val = np.min(values)
    max_val = np.max(values)
    print(f"距离值的最小值: {min_val}, 最大值: {max_val}")  # 添加打印语句检查最小值和最大值
    if max_val == min_val:
        print("所有值都相同。返回零值。")
        return np.zeros_like(values)
    normalized_values = (values - min_val) / (max_val - min_val)
    return normalized_values

def save_sdf_vtk(surf_file, sphe_file, line_file, output_file, log_file):
    """
    读取表面和线数据，计算 SDF 值，归一化并保存到新 .vtk 文件中

    参数:
        surf_file (str): 表面文件路径
        sphe_file (str): 球面文件路径
        line_file (str): 线文件路径
        output_file (str): 输出文件路径
        log_file (str): 日志文件路径
    """
    if not os.path.exists(surf_file) or not os.path.exists(sphe_file) or not os.path.exists(line_file):
        with open(log_file, 'a') as log:
            log.write(f'跳过：{surf_file}, {sphe_file} 或 {line_file} 不存在\n')
        return

    surface_data = read_vtk(sphe_file)
    if surface_data is None:
        with open(log_file, 'a') as log:
            log.write(f'读取球面文件 {sphe_file} 时出错\n')
        return

    line_data = read_vtk(line_file)
    if line_data is None:
        with open(log_file, 'a') as log:
            log.write(f'读取线文件 {line_file} 时出错\n')
        return

    surface_points = surface_data['vertices']
    line_indices = line_data.get('line_indices', None)
    
    if line_indices is None:
        with open(log_file, 'a') as log:
            log.write(f'线文件 {line_file} 中缺少线索引数据\n')
        return

    line_points = extract_line_points(line_indices, surface_points)

    # # 打印表面点和线点的一些示例数据
    # print(f"球面点数: {len(surface_points)}, 线点数: {len(line_points)}")
    # print(f"球面点示例: {surface_points[:5]}")
    # print(f"线点示例: {line_points[:5]}")

    if len(surface_points) == 0 or len(line_points) == 0:
        with open(log_file, 'a') as log:
            log.write(f'错误：表面点或线点数据为空\n')
        return

    sdf_values = compute_sdf(surface_points, line_points)
    if np.all(sdf_values == np.inf):
        with open(log_file, 'a') as log:
            log.write(f'错误：未能为 {sphe_file} 计算有效的 SDF 值\n')
        return

    print(f"SDF 值: {sdf_values[:10]}")

    normalized_sdf_values = normalize_values(sdf_values)
    # print(f"归一化的 SDF 值: {normalized_sdf_values[:10]}")  # 检查归一化的 SDF 值

    original_surface = pv.read(surf_file)
    original_surface['sdf'] = sdf_values
    original_surface['sdf_norm'] = normalized_sdf_values

    try:
        original_surface.save(output_file, binary=False)
    except Exception as e:
        with open(log_file, 'a') as log:
            log.write(f'保存文件 {output_file} 时出错: {e}\n')
        return

if __name__ == '__main__':
    root = './For_Training'
    for subject in os.listdir(root):
        if subject.isdigit():
            for hemi in ['lh']:
                surf_file = os.path.join(root, subject, f'{subject}_recon_40962', 'surf', f'{subject}.{hemi}.InnerSurf.RegByFS.Resp40962.vtk')
                sphe_file = os.path.join(root, subject, f'{subject}_recon_40962', 'surf', f'{subject}.{hemi}.SpheSurf.RegByFS.Resp40962.vtk')
                line_file = os.path.join(root, subject, f'{subject}_gyralnet_island_40962', f'{hemi}_sphere_skelenton_allpoints_final.vtk')
                output_file = os.path.join(root, f'{subject}_{hemi}_sdf.vtk')
                log_file = os.path.join(root, 'sdf.log')
                save_sdf_vtk(surf_file, sphe_file, line_file, output_file, log_file)
