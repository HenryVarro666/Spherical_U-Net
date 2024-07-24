'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-07-16 16:50:41
FilePath: r'\Spherical_U-Net\mv_hemi_npy.py'
'''
# from sphericalunet.utils.vtk import read_vtk
import pyvista as pv
import numpy as np
from nibabel.freesurfer import io as fio
from scipy.spatial import cKDTree
# import torch
import glob
import os
import vtk
import re
from tqdm import tqdm

def read_vtk(in_file):
    """
    Read .vtk POLYDATA file
    
    in_file: string,  the filename
    Out: dictionary, 'vertices', 'faces', 'sulc', 'curv', ...
    """
    polydata = pv.read(in_file)
    vertices = np.array(polydata.points)
    faces = np.array(polydata.faces).reshape(-1, 4)[:, 1:]
    
    data = {'vertices': vertices, 'faces': faces}
    
    point_data = polydata.point_data
    for key, value in point_data.items():
        data[key] = np.array(value)
    
    return data

def compute_sdf(surface_points, line_points):
    tree = cKDTree(line_points)
    distances, _ = tree.query(surface_points)
    return distances

def save_sdf_npy(feature_file, line_file, output_dir, log_file):
    if not os.path.exists(feature_file) or not os.path.exists(line_file):
        with open(log_file, 'a') as log:
            log.write(f'Skipping: {feature_file} or {line_file} does not exist\n')
        return

    surface_data = read_vtk(feature_file)
    line_data = read_vtk(line_file)

    surface_points = surface_data['vertices']
    line_points = line_data['vertices']

    sdf_values = compute_sdf(surface_points, line_points)
    surface_data['sdf'] = sdf_values

    output_filename = os.path.join(output_dir, os.path.basename(feature_file).replace('.vtk', '_sdf.npz'))
    print(f'Saving {output_filename}')
    # np.save(output_filename, surface_data)
    np.savez(output_filename, vertices=surface_data['vertices'], faces=surface_data['faces'], curv=surface_data['curv'], \
             sulc=surface_data['sulc'], sdf=surface_data['sdf'])

# 示例用法
if __name__ == "__main__":
    # 硬编码的路径
    # home_dir = '/media/lab/ef1e5021-01ef-4f9e-9cf7-950095b49199/HCP_fromFenqiang/'
    home_dir = './vtk_files/'
    log_file = 'failed_subjects.log'
    subjects = [subject for subject in os.listdir(home_dir) if re.match(r'^\d+$', subject)]
    
    for subject in tqdm(subjects, desc="Processing subjects"):
        print(f'Processing {subject}')
        for hemi in ['lh', 'rh']:
            feature_file = os.path.join(home_dir, subject, subject+'_recon_40962','surf', f'{subject}.{hemi}.InnerSurf.RegByFS.Resp40962.vtk')
            line_file = os.path.join(home_dir, subject, subject+'_gyralnet_island_40962', f'{hemi}_surf_skelenton_allpoints_final.vtk')
            # output_dir = '/home/lab/Documents/Spherical_Dataset/'
            output_dir = './output_dir/'
            save_sdf_npy(feature_file, line_file, output_dir, log_file)