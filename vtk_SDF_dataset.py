'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-07-14 16:48:14
FilePath: /Spherical_U-Net/vtk_2_dataloader.py
'''
# from sphericalunet.utils.vtk import read_vtk
import pyvista
import numpy as np
from nibabel.freesurfer import io as fio

import torch
import glob
import os

import vtk


# # Example 1
# def delete_this():
#     """
#     This function reads geometry and morph data from files and saves it in a VTK file format.
#     It performs the following steps:
#     1. Reads points and faces from a geometry file.
#     2. Reads sulc, curv, and thickness data from morph files.
#     3. Creates a PolyData object using the points and faces.
#     4. Assigns sulc, curv, and thickness data to the PolyData object.
#     5. Saves the PolyData object in VTK file format.

#     Returns:
#     - None
#     """
#     points, faces = fio.read_geometry('/media/jialec/DATA/HCP_download_for_weiyan/100206_3T_Structural_preproc_extended/100206/T1w/100206/surf/lh.sphere')
#     sulc = fio.read_morph_data('/media/jialec/DATA/HCP_download_for_weiyan/100206_3T_Structural_preproc_extended/100206/T1w/100206/surf/lh.sulc')
#     curv = fio.read_morph_data('/media/jialec/DATA/HCP_download_for_weiyan/100206_3T_Structural_preproc_extended/100206/T1w/100206/surf/lh.curv')
#     thickness = fio.read_morph_data('/media/jialec/DATA/HCP_download_for_weiyan/100206_3T_Structural_preproc_extended/100206/T1w/100206/surf/lh.thickness')

#     # points, faces = fio.read_geometry('/usr/local/freesurfer/7.2.0/subjects/fsaverage/surf/lh.sphere')
#     sulc = fio.read_morph_data('/usr/local/freesurfer/7.2.0/subjects/fsaverage/surf/lh.sulc')
#     curv = fio.read_morph_data('/usr/local/freesurfer/7.2.0/subjects/fsaverage/surf/lh.curv')
#     # par_fs = fio.read_annot('/usr/local/freesurfer/7.2.0/subjects/fsaverage/label/lh.aparc.a2005s.annot')
#     # par_fs_2009 = fio.read_annot('/usr/local/freesurfer/7.2.0/subjects/fsaverage/label/lh.aparc.a2009s.annot')

#     faces = np.concatenate((np.ones((faces.shape[0], 1))*3, faces), axis=-1).astype(np.int32)
#     surf = pyvista.PolyData(points, faces)
#     surf['sulc'] = sulc
#     surf['curv'] = curv
#     surf['thickness'] = thickness
#     # surf['par_fs'] = par_fs
#     # surf['par_fs_2009'] = par_fs_2009
#     # surf.save('./delete_this.vtk', binary=False)
#     surf.save('./delete_this_hcp.fsaverage.vtk', binary=False)
#     print("True")
#     return

# # Example 2
# def read_vtk(in_file):
#     """
#     Read .vtk POLYDATA file
    
#     in_file: string,  the filename
#     Out: dictionary, 'vertices', 'faces', 'curv', 'sulc', ...
#     """
#     polydata = pyvista.read(in_file)
    
#     # n_faces = polydata.n_faces

#     n_faces = polydata.n_faces_strict

#     # import pdb; pdb.set_trace()
#     vertices = np.array(polydata.points)  # get vertices coordinate

#     # only for triangles polygons data
#     faces = np.array(polydata.GetPolys().GetData())  # get faces connectivity
#     assert len(faces)/4 == n_faces, "faces number is wrong!"
#     faces = np.reshape(faces, (n_faces,4))
    
#     data = {'vertices': vertices,
#             'faces': faces
#             }
    
#     point_data = polydata.point_data
#     for key, value in point_data.items():
#         if value.dtype == 'uint32':
#             data[key] = np.array(value).astype(np.int64)
#         elif  value.dtype == 'uint8':
#             data[key] = np.array(value).astype(np.int32)
#         else:
#             data[key] = np.array(value)

#     return data

def read_vtk(file_path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def vtk_to_numpy(vtk_data):
    points = vtk_data.GetPoints()
    num_points = points.GetNumberOfPoints()
    features = np.zeros((num_points, 3))

    for i in range(num_points):
        features[i, :] = points.GetPoint(i)

    return features



class BrainDatasetSDF(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_files = sorted(glob.glob(os.path.join(data_dir, '*_sdf.npy')))
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = self.data_files[idx]
        data = np.load(data_path)
        features, sdf = data[:, :-1], data[:, -1]

        if self.transform:
            features, sdf = self.transform(features, sdf)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(sdf, dtype=torch.float32)
