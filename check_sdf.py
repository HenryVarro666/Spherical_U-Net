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

def save_sdf_vtk(surf_file, line_file, output_dir, log_file):
    if not os.path.exists(surf_file) or not os.path.exists(line_file):
        with open(log_file, 'a') as log:
            log.write(f'Skipping: {surf_file} or {line_file} does not exist\n')
        return

    surface_data = read_vtk(surf_file)
    line_data = read_vtk(line_file)

    surface_points = surface_data['vertices']
    line_points = line_data['vertices']

    sdf_values = compute_sdf(surface_points, line_points)
    surface_data['sdf'] = sdf_values

    # output_filename = os.path.join(output_dir, os.path.basename(surf_file).replace('.vtk', '_sdf.npz'))
    
    # root = '/mnt/d/Spherical_U-Net/vtk_files'
    # output_dir = os.path.join(root, 'sdf.vtk')
    # inner.save(file_path.replace('.withGrad.164k_fsaverage.flip.rescale.Sphere.vtk', '.withGrad.164k_fsaverage.flip.rescale.Inner.vtk'), binary=False)
    surface_data.save(output_dir, binary=False)

if __name__ == '__main__':
    root = '/mnt/d/Spherical_U-Net/vtk_files'
    for subject in os.listdir(root):
        if subject.isdigit():
            for hemi in ['lh', 'rh']:
                surf_files = os.path.join(root, f'{subject}_recon_40962', 'surf', f'{subject}.{hemi}.InnerSurf.RegByFS.Resp40962.vtk')
                line_files = os.path.join(root, f'{subject}_gyralnet_island', f'{hemi}_surf_skelenton_allpoints_final.vtk')
                output_dir = os.path.join(root, f'{subject}_sdf.vtk')
                log_file = os.path.join(root, f'{subject}_sdf.log')
                save_sdf_vtk(surf_files, line_files, output_dir, log_file)

