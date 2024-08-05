'''
Author: HenryVarro666 1504517223@qq.com
Date: 2024-08-05 09:22:58
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-08-05 15:01:15
FilePath: \Spherical_U-Net\predict_cla_reg.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import argparse
import numpy as np
import os
import cv2  # 添加OpenCV库
from model import Unet_40k, Unet_160k, Unet_40k_batch_2
from sphericalunet.utils.vtk import read_vtk, write_vtk, resample_label
from sphericalunet.utils.interp_numpy import resampleSphereSurf
from torch.nn.functional import sigmoid

def inference(curv, sulc, model, device):
    feats = torch.cat((curv, sulc), 1)
    feat_max = torch.tensor([1.2, 13.7], device=device)
    feats = feats / feat_max
    with torch.no_grad():
        feats = feats.unsqueeze(0)  # Add batch dimension
        predict_reg, predict_cla = model(feats)
        predict_cla = torch.argmax(predict_cla, dim=1)
    return predict_reg.cpu().numpy(), predict_cla.cpu().numpy()

# def connected_component_analysis(pred):
#     # 将预测结果转换为二值图像
#     pred_binary = np.array(pred > 0.5, dtype=np.uint8)
#     # 使用OpenCV的connectedComponents函数进行连通域分析
#     num_labels, labels_im = cv2.connectedComponents(pred_binary)
#     # num_labels, labels_im = cv2.connectedComponents(pred)

#     return num_labels, labels_im

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
    
    model = Unet_40k_batch_2(2, 1, 3) if level == '7' else Unet_160k(2, 1)
    model_path = f'trained_models_cla_reg/Unet_{"40k_batch_2_1_final.pkl" if level == "7" else "160k_curv_sulc.pkl"}'
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

    pred_reg, pred_cla = inference(curv, sulc, model, device)

    pred_reg = pred_reg.squeeze()  # (N, 1) -> (N)
    pred_cla = pred_cla.squeeze()  # (N, 1) -> (N)

    orig_surf['gyralnet_prediction_reg'] = pred_reg
    orig_surf['gyralnet_prediction_cla'] = pred_cla

    write_vtk(orig_surf, out_file)
