'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-07-14 17:02:09
FilePath: /Spherical_U-Net/vtk_2_dataloader.py
'''
# from sphericalunet.utils.vtk import read_vtk
import pyvista
import numpy as np
from nibabel.freesurfer import io as fio
from scipy.spatial import cKDTree

import torch
import glob
import os
import vtk

def read_vtk(in_file):
    """
    Read .vtk POLYDATA file
    
    in_file: string,  the filename
    Out: dictionary, 'vertices', 'faces', 'sulc', 'curv', ...
    """
    polydata = pyvista.read(in_file)
    
    n_faces = polydata.n_faces_strict
    vertices = np.array(polydata.points)  # get vertices coordinate
    faces = np.array(polydata.faces).reshape(-1, 4)[:, 1:]  # get faces connectivity
    
    data = {'vertices': vertices,
            'faces': faces}
    
    point_data = polydata.point_data
    for key, value in point_data.items():
        data[key] = np.array(value)
    
    return data

def write_vtk(in_dic, file, binary=True):
    """
    Write .vtk POLYDATA file
    
    in_dic: dictionary, vtk data
    file: string, output file name
    """
    vertices = in_dic['vertices']
    faces = in_dic['faces']
    polydata = pv.PolyData(vertices, faces)
    for key, value in in_dic.items():
        if key not in ['vertices', 'faces']:
            polydata.point_data[key] = value
    polydata.save(file, binary=binary)

def compute_sdf(surface_points, line_points):
    tree = cKDTree(line_points)
    distances, _ = tree.query(surface_points)
    return distances

def save_sdf_npy(feature_file, line_file, output_dir):
    surface_data = read_vtk(feature_file)
    line_data = read_vtk(line_file)

    surface_points = surface_data['vertices']
    line_points = line_data['vertices']

    sdf_values = compute_sdf(surface_points, line_points)

    # 将 SDF 值添加为特征数据
    surface_data['sdf'] = sdf_values

    # 保存为 npy 文件
    output_filename = os.path.join(output_dir, os.path.basename(feature_file).replace('.vtk', '_sdf.npy'))
    np.save(output_filename, surface_data)

# 示例用法
feature_file = 'path/to/surface.vtk'
line_file = 'path/to/line.vtk'
output_dir = 'path/to/output_dir'

save_sdf_npy(feature_file, line_file, output_dir)



# class BrainDatasetSDF(torch.utils.data.Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.data_files = sorted(glob.glob(os.path.join(data_dir, '*_sdf.npy')))
#         self.transform = transform

#     def __len__(self):
#         return len(self.data_files)

#     def __getitem__(self, idx):
#         data_path = self.data_files[idx]
#         data = np.load(data_path)
#         features, sdf = data[:, :-1], data[:, -1]

#         if self.transform:
#             features, sdf = self.transform(features, sdf)

#         return torch.tensor(features, dtype=torch.float32), torch.tensor(sdf, dtype=torch.float32)
