'''
Author: Chao Cao 1504517223@qq.com
Date: 2024-07-16 14:06:33
LastEditors: Chao Cao 1504517223@qq.com
LastEditTime: 2024-07-16 14:16:03
FilePath: /Spherical_U-Net/tmp.py
'''
# import torch
# print(torch.cuda.is_available())
# print(torch.version.cuda)
# print(torch.cuda.get_device_name(0))

# import torch
# print(torch.__version__)
# print(torch.backends.mps.is_available())
# print(torch.backends.mps.is_built())

# import torch
# print(f"PyTorch version: {torch.__version__}")
# print(f"MPS available: {torch.backends.mps.is_available()}")
# print(f"MPS built: {torch.backends.mps.is_built()}")

import torch

# 模拟传递参数
class Args:
    device = 'mps'  # 这里设置为 'mps', 你可以改成 'GPU' 或 'CPU' 进行测试

args = Args()

device = args.device

if device == 'mps':
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        torch.backends.mps.enable_non_blocking = True
    else:
        print('MPS device is not available on this system, falling back to CPU.')
        device = torch.device('cpu')
elif device == 'GPU':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        print('CUDA is not available on this system, falling back to CPU.')
        device = torch.device('cpu')
elif device == 'CPU':
    device = torch.device('cpu')
else:
    raise NotImplementedError('Only support GPU, CPU, or MPS device')

print(f'Using device: {device}')
