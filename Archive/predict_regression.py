# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import torch
# import argparse
# import numpy as np
# import os
# from model import Unet_40k, Unet_160k
# from sphericalunet.utils.vtk import read_vtk, write_vtk, resample_label
# from sphericalunet.utils.interp_numpy import resampleSphereSurf

# def inference(curv, sulc, model, device):
#     feats = torch.cat((curv, sulc), 1)
#     feat_max = torch.tensor([1.2, 13.7], device=device)
#     feats = feats / feat_max
#     with torch.no_grad():
#         prediction = model(feats)
#     return prediction.cpu().numpy()

# if __name__ == "__main__":    
#     parser = argparse.ArgumentParser(description='Predict the parcellation maps with 36 regions from the input surfaces',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--hemisphere', '-hemi', default='left',
#                         choices=['left', 'right'], 
#                         help="Specify the hemisphere for parcellation, left or right.")
#     parser.add_argument('--level', '-l', default='7',
#                         choices=['7', '8'],
#                         help="Specify the level of the surfaces\' resolution. Generally, level 7 with 40962 vertices is sufficient, level 8 with 163842 vertices is more accurate but slower.")
#     parser.add_argument('--input', '-i', metavar='INPUT',
#                         help='filename of input surface')
#     parser.add_argument('--output', '-o',  default='[input].parc.vtk', metavar='OUTPUT',
#                         help='Filename of output surface.')
#     parser.add_argument('--device', default='GPU', choices=['GPU', 'CPU'], 
#                         help='the device for running the model.')

#     args = parser.parse_args()
#     in_file = args.input
#     out_file = args.output
#     hemi = args.hemisphere
#     level = args.level
   
#     device = torch.device('cuda:0' if args.device == 'GPU' else 'cpu')

#     if not in_file:
#         raise ValueError('Input filename is required')
#     if out_file == '[input].parc.vtk':
#         out_file = in_file.replace('.vtk', '.parc.vtk')
    
#     model = Unet_40k(2, 1) if level == '7' else Unet_160k(2, 1)
#     model_path = f'trained_models/Unet_{"40k_1.pkl" if level == "7" else "160k_curv_sulc.pkl"}'
#     n_vertices = 40962 if level == '7' else 163842
    
#     model.to(device)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
    
#     template = read_vtk(f'neigh_indices/sphere_{n_vertices}_rotated_0.vtk')

#     orig_surf = read_vtk(in_file)
#     curv_temp = orig_surf['curv']
#     if len(curv_temp) != n_vertices:
#         sucu = resampleSphereSurf(orig_surf['vertices'], template['vertices'], 
#                                   np.concatenate((orig_surf['sulc'][:, np.newaxis], 
#                                                   orig_surf['curv'][:, np.newaxis]), axis=1))
#         sulc, curv = sucu[:, 0], sucu[:, 1]
#     else:
#         curv, sulc = orig_surf['curv'][:n_vertices], orig_surf['sulc'][:n_vertices]

#     curv = torch.from_numpy(curv).unsqueeze(1).to(device)
#     sulc = torch.from_numpy(sulc).unsqueeze(1).to(device)
    
#     pred = inference(curv, sulc, model, device)
    
#     orig_surf['gyralnet_prediction'] = pred
#     write_vtk(orig_surf, out_file)


###############################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import argparse
import numpy as np
import os
from model import Unet_40k, Unet_160k
from sphericalunet.utils.vtk import read_vtk, write_vtk, resample_label
from sphericalunet.utils.interp_numpy import resampleSphereSurf

def inference(curv, sulc, model, device):
    feats = torch.cat((curv, sulc), 1)
    feat_max = torch.tensor([1.2, 13.7], device=device)
    feats = feats / feat_max
    with torch.no_grad():
        prediction = model(feats)
    return prediction.cpu().numpy()

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Predict the parcellation maps with 36 regions from the input surfaces',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hemisphere', '-hemi', default='left',
                        choices=['left', 'right'], 
                        help="Specify the hemisphere for parcellation, left or right.")
    parser.add_argument('--level', '-l', default='7',
                        choices=['7', '8'],
                        help="Specify the level of the surfaces' resolution. Generally, level 7 with 40962 vertices is sufficient, level 8 with 163842 vertices is more accurate but slower.")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='filename of input surface')
    parser.add_argument('--output', '-o',  default='[input].parc.vtk', metavar='OUTPUT',
                        help='Filename of output surface.')
    parser.add_argument('--device', default='GPU', choices=['GPU', 'CPU'], 
                        help='the device for running the model.')

    args = parser.parse_args()
    in_file = args.input
    out_file = args.output
    hemi = args.hemisphere
    level = args.level
   
    device = torch.device('cuda:0' if args.device == 'GPU' else 'cpu')

    if not in_file:
        raise ValueError('Input filename is required')
    if out_file == '[input].parc.vtk':
        out_file = in_file.replace('.vtk', '.parc.vtk')
    
    model = Unet_40k(2, 1) if level == '7' else Unet_160k(2, 1)
    # model_path = f'trained_models/Unet_{"40k_1.pkl" if level == "7" else "160k_curv_sulc.pkl"}'

    model_path = f'trained_models_3/Unet_{"40k_1_final.pkl" if level == "7" else "160k_curv_sulc.pkl"}'
    n_vertices = 40962 if level == '7' else 163842
    
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    template = read_vtk(f'neigh_indices/sphere_{n_vertices}_rotated_0.vtk')

    orig_surf = read_vtk(in_file)
    curv_temp = orig_surf['curv']
    if len(curv_temp) != n_vertices:
        sucu = resampleSphereSurf(orig_surf['vertices'], template['vertices'], 
                                  np.concatenate((orig_surf['sulc'][:, np.newaxis], 
                                                  orig_surf['curv'][:, np.newaxis]), axis=1))
        sulc, curv = sucu[:, 0], sucu[:, 1]
    else:
        curv, sulc = orig_surf['curv'][:n_vertices], orig_surf['sulc'][:n_vertices]

    curv = torch.from_numpy(curv).unsqueeze(1).to(device)
    sulc = torch.from_numpy(sulc).unsqueeze(1).to(device)
    
    pred = inference(curv, sulc, model, device)
    
    # 找到前 5% 最低值的阈值
    # threshold = np.percentile(pred, 5)

    threshold = np.percentile(pred, 0)

    
    # 修改预测结果，使得前 5% 最低值设为 0，其他值设为 1
    # binary_pred = np.where(pred <= 0, 0, 1).flatten()

    binary_pred = np.where(pred <= threshold, 0, 1).flatten()
    
    # 更新原始表面数据
    orig_surf['gyralnet_prediction'] = binary_pred

    write_vtk(orig_surf, out_file)

