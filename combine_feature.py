'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: Chao Cao 1504517223@qq.com
LastEditTime: 2024-07-16 12:41:59
FilePath: /Spherical_U-Net/combine_feature.py
'''

import pyvista
import numpy as np
from nibabel.freesurfer import io as fio
import os
import re

def combine():
    surf = pyvista.read(surf_path)
    
    curv_value = fio.read_morph_data(curv_path)
    sulc_value = fio.read_morph_data(sulc_path)

    surf['curv'] = curv_value
    surf['sulc'] = sulc_value
    surf.save(surf_path.replace('.InnerSurf.RegByFS.Resp40962.vtk', '.InnerSurf.RegByFS.Resp40962.vtk'), binary=False)
    print(f'{subject} {hemi} Done')
    # surf.save('/mnt/d/Spherical_U-Net/examples/new.vtk', binary=False)

if __name__ == '__main__':
    root_dir = '/work/users/c/h/chaocao/For_Training'
    for subject in os.listdir(root_dir):
        if re.match(r'^\d+$', subject):
            for hemi in ['lh', 'rh']:
                surf_path = os.path.join(root_dir, subject, subject +'_recon_40962', 'surf', f'{subject}.{hemi}.InnerSurf.RegByFS.Resp40962.vtk')
                curv_path = os.path.join(root_dir, subject, subject +'_recon_40962','surf', f'{hemi}.curv')
                sulc_path = os.path.join(root_dir, subject, subject +'_recon_40962','surf', f'{hemi}.sulc')
                combine()
                print('Done')