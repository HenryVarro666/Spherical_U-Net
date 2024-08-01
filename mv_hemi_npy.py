## Backup: save_sdf_npy.py

# # from sphericalunet.utils.vtk import read_vtk
# import pyvista as pv
# import numpy as np
# from nibabel.freesurfer import io as fio
# from scipy.spatial import cKDTree
# # import torch
# import glob
# import os
# import vtk
# import re
# from tqdm import tqdm

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
#     tree = cKDTree(line_points)
#     distances, _ = tree.query(surface_points)
#     return distances

# def save_sdf_npy(feature_file, line_file, output_dir, log_file):
#     if not os.path.exists(feature_file) or not os.path.exists(line_file):
#         with open(log_file, 'a') as log:
#             log.write(f'Skipping: {feature_file} or {line_file} does not exist\n')
#         return

#     surface_data = read_vtk(feature_file)
#     line_data = read_vtk(line_file)

#     surface_points = surface_data['vertices']
#     line_points = line_data['vertices']

#     sdf_values = compute_sdf(surface_points, line_points)
#     surface_data['sdf'] = sdf_values

#     output_filename = os.path.join(output_dir, os.path.basename(feature_file).replace('.vtk', '_sdf.npz'))
#     print(f'Saving {output_filename}')
#     # np.save(output_filename, surface_data)
#     np.savez(output_filename, vertices=surface_data['vertices'], faces=surface_data['faces'], curv=surface_data['curv'], \
#              sulc=surface_data['sulc'], sdf=surface_data['sdf'])

# # 示例用法
# if __name__ == "__main__":
#     # 硬编码的路径
#     home_dir = '/work/users/c/h/chaocao/HCP_fromFenqiang'
#     log_file = 'failed_subjects.log'
#     subjects = [subject for subject in os.listdir(home_dir) if re.match(r'^\d+$', subject)]
    
#     # subjects = subjects[:60]

#     for subject in tqdm(subjects, desc="Processing subjects"):
#         print(f'Processing {subject}')
#         for hemi in ['lh', 'rh']:
#             feature_file = os.path.join(home_dir, subject, subject+'_recon_40962','surf', f'{subject}.{hemi}.InnerSurf.RegByFS.Resp40962.vtk')
#             line_file = os.path.join(home_dir, subject, subject+'_gyralnet_island_40962', f'{hemi}_surf_skelenton_allpoints_final.vtk')
#             output_dir = '/work/users/c/h/chaocao/npz_dataset'
#             save_sdf_npy(feature_file, line_file, output_dir, log_file)

############################################################################################################

'''
Author: HenryVarro666 1504517223@qq.com
Date: 2024-07-15 08:23:17
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-07-18 17:33:35
FilePath: \Spherical_U-Net\mv_hemi_npy.py
'''
import os
import shutil
import glob
from sklearn.model_selection import KFold

def create_folds(data_dir, num_folds=6):
    # 创建folds目录
    lh_dir = os.path.join(data_dir, 'lh')
    rh_dir = os.path.join(data_dir, 'rh')
    
    # 获取所有 .npy 文件
    lh_files = sorted(glob.glob(os.path.join(data_dir, '*.lh.*.npz')))
    rh_files = sorted(glob.glob(os.path.join(data_dir, '*.rh.*.npz')))
    # Debug: Print the number of files found
    print(f"Found {len(lh_files)} lh files and {len(rh_files)} rh files.")

    # Check if the number of files is less than the number of folds
    if len(lh_files) < num_folds or len(rh_files) < num_folds:
        print("Error: Not enough files to create the specified number of folds.")
        return
    
    # 创建KFold对象
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # 创建lh和rh的fold目录
    for fold in range(1, num_folds + 1):
        os.makedirs(os.path.join(lh_dir, f'fold{fold}'), exist_ok=True)
        os.makedirs(os.path.join(rh_dir, f'fold{fold}'), exist_ok=True)
    
    # 将lh文件分成folds
    for fold_idx, (_, fold_indices) in enumerate(kf.split(lh_files), 1):
        for idx in fold_indices:
            shutil.move(lh_files[idx], os.path.join(lh_dir, f'fold{fold_idx}', os.path.basename(lh_files[idx])))
    
    # 将rh文件分成folds
    for fold_idx, (_, fold_indices) in enumerate(kf.split(rh_files), 1):
        for idx in fold_indices:
            shutil.move(rh_files[idx], os.path.join(rh_dir, f'fold{fold_idx}', os.path.basename(rh_files[idx])))

    print("Files have been divided into 6 folds for both lh and rh.")

# 示例用法
if __name__ == "__main__":
    # 硬编码的源文件夹路径
    data_dir = './Test3'
    
    create_folds(data_dir)
