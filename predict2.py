#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import argparse
import numpy as np
import os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from model import Unet_40k, Unet_160k
from sphericalunet.utils.vtk import read_vtk, write_vtk, resample_label
from sphericalunet.utils.interp_numpy import resampleSphereSurf
from torch.nn.functional import sigmoid

def inference(curv, sulc, model, device):
    feats = torch.cat((curv, sulc), 1)
    feat_max = torch.tensor([1.2, 13.7], device=device)
    feats = feats / feat_max
    with torch.no_grad():
        feats = feats.unsqueeze(0)  # Add batch dimension
        prediction = model(feats)
        prediction = sigmoid(prediction)
    return prediction.cpu().numpy()

def build_adjacency_matrix(vertices, faces):
    n_vertices = vertices.shape[0]
    row_indices = []
    col_indices = []
    for face in faces:
        for i in range(len(face)):
            for j in range(i + 1, len(face)):
                row_indices.append(face[i])
                col_indices.append(face[j])
                row_indices.append(face[j])
                col_indices.append(face[i])
    data = np.ones(len(row_indices), dtype=np.int32)
    adjacency_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n_vertices, n_vertices))
    return adjacency_matrix

def connected_component_analysis_surface(vertices, faces, pred):
    adjacency_matrix = build_adjacency_matrix(vertices, faces)
    num_labels, labels = connected_components(csgraph=adjacency_matrix, directed=False, return_labels=True)
    
    # 将所有连通域标签合并成一个
    labels[labels != 0] = 1  # 将所有非背景区域标记为1
    return 2, labels  # 返回两个标签：背景和一个连通域

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

    pred_prob = pred.squeeze()  # (N, 1) -> (N)
    pred = np.array(pred_prob > 0.5, dtype=np.int32)

    vertices = orig_surf['vertices']
    faces = orig_surf['faces']  # 假设faces包含表面的面信息

    num_labels, labels_im = connected_component_analysis_surface(vertices, faces, pred)

    orig_surf['gyralnet_prediction'] = pred
    orig_surf['gyralnet_prediction_prob'] = pred_prob
    orig_surf['gyralnet_cc_labels'] = labels_im

    write_vtk(orig_surf, out_file)

    print(f"Number of connected components: {num_labels}")
    print(f"Labels array:\n {labels_im}")
