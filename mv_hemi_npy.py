'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-07-14 18:41:01
FilePath: /Spherical_U-Net/mv_hemi_npy.py
'''
import os
import shutil
from tqdm import tqdm

def move_files_by_hemi(source_dir):
    # 定义目标文件夹路径
    lh_dir = os.path.join(source_dir, 'lh')
    rh_dir = os.path.join(source_dir, 'rh')
    
    # 创建目标文件夹，如果不存在
    os.makedirs(lh_dir, exist_ok=True)
    os.makedirs(rh_dir, exist_ok=True)
    
    # 获取源文件夹中的所有文件
    files = os.listdir(source_dir)
    
    # 遍历源文件夹中的文件，并显示进度条
    for filename in tqdm(files, desc="Moving files"):
        # 检查文件是否是 .npy 文件
        if filename.endswith('_sdf.npy'):
            # 根据文件名中的 'lh' 或 'rh' 移动文件
            if 'lh' in filename:
                shutil.move(os.path.join(source_dir, filename), os.path.join(lh_dir, filename))
            elif 'rh' in filename:
                shutil.move(os.path.join(source_dir, filename), os.path.join(rh_dir, filename))
    
    print("Files have been moved to respective folders.")

# 示例用法
if __name__ == "__main__":
    # 硬编码的源文件夹路径
    source_dir = '/mnt/d/Spherical_U-Net/Test'
    
    move_files_by_hemi(source_dir)
