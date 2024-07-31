'''
Author: HenryVarro666 1504517223@qq.com
Date: 2024-07-31 17:03:49
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-07-31 17:07:10
FilePath: /Spherical_U-Net/val_dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import glob
import os

class BrainSphereInspector:
    def __init__(self, *data_dirs):
        self.data_files = []
        for data_dir in data_dirs:
            files = sorted(glob.glob(os.path.join(data_dir, '*_linemask_skeleton.npz')))
            self.data_files.extend(files)

    def check_features(self):
        problem_files = []
        for file in self.data_files:
            data = np.load(file, allow_pickle=True)

            # 提取特征
            sulc = data['sulc']
            curv = data['curv']
            feats = np.stack((sulc, curv), axis=1)

            # 对每个特征独立归一化
            feat_max = np.max(feats, axis=0, keepdims=True)
            feats = feats / feat_max

            # 检查是否存在NaN或Inf值
            if np.any(np.isnan(feats)) or np.any(np.isinf(feats)):
                problem_files.append(file)
                print(f"Problem found in file: {file}")

        if not problem_files:
            print("No issues found in any files.")
        else:
            print(f"Total problem files: {len(problem_files)}")

if __name__ == "__main__":
    # 指定数据文件夹路径
    fold1 = './Test4/lh/fold1'
    fold2 = './Test4/lh/fold2'
    fold3 = './Test4/lh/fold3'
    fold4 = './Test4/lh/fold4'
    fold5 = './Test4/lh/fold5'
    fold6 = './Test4/lh/fold6'

    inspector = BrainSphereInspector(fold1, fold2, fold3, fold4, fold5, fold6)
    inspector.check_features()
