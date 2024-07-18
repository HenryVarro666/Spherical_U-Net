'''
Author: HenryVarro666 1504517223@qq.com
Date: 2024-07-15 08:23:17
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-07-18 16:46:41
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
    data_dir = '/mnt/d/Spherical_U-Net/Test'
    
    create_folds(data_dir)
