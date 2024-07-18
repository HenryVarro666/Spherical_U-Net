#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:21:19 2018

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com
"""

import torch
import torch.nn as nn
import torchvision
import scipy.io as sio 
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter # type: ignore
# writer = SummaryWriter('log/a')

from model import Unet_40k, Unet_160k

################################################################
""" hyper-parameters """
# 指定使用第一个 GPU 进行训练。如果有多个 GPU，可以根据需要更改设备索引
cuda = torch.device('cuda:0')
# 每个批次中包含的样本数。在训练和验证过程中，将数据分割成多个批次处理
batch_size = 1
fold = 1 # 1,2,3 
model_name = 'Unet_40k'  # 'Unet_40k', 'Unet_160k'
up_layer = 'upsample_interpolation' # 'upsample_interpolation', 'upsample_fixindex' 
in_channels = 2
out_channels = 36
learning_rate = 0.001
# 动量项，用于优化器中，帮助加速收敛
momentum = 0.99
# 权重衰减（L2 正则化）系数，用于防止过拟合
weight_decay = 0.0001
################################################################


class BrainSphere(torch.utils.data.Dataset):
    def __init__(self, *data_dirs):
        self.data_files = []
        for data_dir in data_dirs:
            files = sorted(glob.glob(os.path.join(data_dir, '*_sdf.npy')))
            self.data_files.extend(files) 

    def __getitem__(self, index):
        file = self.data_files[index]
        data = np.load(file, allow_pickle=True).item()
        
        # Extract features
        sulc = data['sulc']
        curv = data['curv']
        feats = np.stack((sulc, curv), axis=1)
        
        # Normalize features
        feat_max = np.max(feats, axis=0)
        for i in range(feats.shape[1]):
            feats[:, i] = feats[:, i] / feat_max[i]
        
        # Extract labels
        label = data['sdf']
        label = np.squeeze(label) - 1
        
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data_files)

fold1 = '/mnt/d/Spherical_U-Net/Test/lh/fold1'
fold2 = '/mnt/d/Spherical_U-Net/Test/lh/fold2'
fold3 = '/mnt/d/Spherical_U-Net/Test/lh/fold3'
fold4 = '/mnt/d/Spherical_U-Net/Test/lh/fold4'
fold5 = '/mnt/d/Spherical_U-Net/Test/lh/fold5'
fold6 = '/mnt/d/Spherical_U-Net/Test/lh/fold6'

if fold == 1:
    train_dataset = BrainSphere(fold3,fold6,fold2,fold5)
    val_dataset = BrainSphere(fold1)
elif fold == 2:
    train_dataset = BrainSphere(fold1,fold4,fold3,fold6)
    val_dataset = BrainSphere(fold2)
elif fold == 3:
    train_dataset = BrainSphere(fold1,fold4,fold2,fold5)
    val_dataset = BrainSphere(fold3)
else:
    raise NotImplementedError('fold name is wrong!')

print(f'Number of samples in train_dataset: {len(train_dataset)}')
print(f'Number of samples in val_dataset: {len(val_dataset)}')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)



############################################################################################################

# if model_name == 'Unet_40k':
#     model = Unet_40k(in_ch=in_channels, out_ch=out_channels)
# elif model_name == 'Unet_160k':
#     model = Unet_160k(in_ch=in_channels, out_ch=out_channels)
# else:
#     raise NotImplementedError('model name is wrong!')


# # model.parameters()：返回模型的所有参数。
# # x.numel()：返回参数张量 x 中的元素总数。
# # sum(x.numel() for x in model.parameters())：计算模型所有参数的元素总数，即模型的总参数数量。
# print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
# # 将模型移动到指定的 GPU 上进行计算
# model.cuda(cuda)
# # 使用交叉熵损失函数 nn.CrossEntropyLoss
# criterion = nn.CrossEntropyLoss()
# # 使用 Adam 优化器 torch.optim.Adam 来优化模型参数
# # lr=learning_rate：设置学习率
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# # 使用学习率调度器 torch.optim.lr_scheduler.ReduceLROnPlateau
# # optimizer：与之关联的优化器。
# # 'max'：监控的指标，如果指标没有改善，学习率将减少。
# # factor=0.2：当指标停止改善时，将学习率减少到原来的 20%。
# # patience=1：如果指标没有改善超过 patience 个周期，则减少学习率。
# # verbose=True：启用详细输出。
# # threshold=0.0001：用于确定何时降低学习率的阈值。
# # threshold_mode='rel'：使用相对变化来确定何时降低学习率。
# # min_lr=0.000001：设置学习率的下限。
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=0.000001)


# def train_step(data, target):
#     # 设置模型为训练模式
#     model.train()
#     # 将数据和标签移动到GPU
#     data, target = data.cuda(cuda), target.cuda(cuda)

#     # 前向传播
#     prediction = model(data)
#     # 计算损失
#     loss = criterion(prediction, target)
#     # 清零梯度
#     optimizer.zero_grad()
#     # 反向传播
#     loss.backward()
#     # 更新模型参数
#     optimizer.step()
#     # 返回损失值
#     return loss.item()


# def compute_dice(pred, gt):
#     # 使用 .cpu().numpy() 方法将预测结果 pred 和真实标签 gt 
#     # 从 GPU 内存移动到 CPU 内存，并转换为 NumPy 数组。
#     pred = pred.cpu().numpy()
#     gt = gt.cpu().numpy()
    
#     # 创建一个长度为 36 的零数组，用于存储每个类别的 Dice 系数。假设有 36 个类别。
#     dice = np.zeros(36)
#     for i in range(36):
#         # 使用 np.where(gt == i)[0] 找到真实标签中类别 i 的索引。
#         gt_indices = np.where(gt == i)[0]
#         # 使用 np.where(pred == i)[0] 找到预测结果中类别 i 的索引。
#         pred_indices = np.where(pred == i)[0]
#         # 使用 np.intersect1d(gt_indices, pred_indices) 找到真实标签和预测结果中类别 i 的交集索引。
#         # 计算类别 i 的 Dice 系数
#         dice[i] = 2 * len(np.intersect1d(gt_indices, pred_indices))/(len(gt_indices) + len(pred_indices))
#     return dice


# def val_during_training(dataloader):
#     # 将模型设置为评估模式。这会禁用 dropout 层和 batch normalization 层的训练行为
#     model.eval()

#     # 创建一个零数组 dice_all，用于存储每个批次和每个类别的 Dice 系数。假设有 36 个类别，len(dataloader) 是验证数据集的批次数。
#     dice_all = np.zeros((len(dataloader),36))
#     # 使用 enumerate 遍历 dataloader 中的每个批次
#     for batch_idx, (data, target) in enumerate(dataloader):
#         # data.squeeze() 和 target.squeeze()：移除维度为 1 的维度。
#         data = data.squeeze()
#         target = target.squeeze()
#         # 将数据和标签移动到 GPU 上
#         data, target = data.cuda(cuda), target.cuda(cuda)
#         # with torch.no_grad()：在上下文管理器 torch.no_grad() 中进行前向传播，禁用梯度计算，以减少内存使用和加速计算。
#         with torch.no_grad():
#             prediction = model(data)
#         # 使用 prediction.max(1)[1] 找到预测结果中每个像素的最大值索引，即预测的类别。    
#         prediction = prediction.max(1)[1]
#         # 计算当前批次的 Dice 系数，并存储在 dice_all 数组中。
#         dice_all[batch_idx,:] = compute_dice(prediction, target)

#     return dice_all


# train_dice = [0, 0, 0, 0, 0]
# # 循环100个训练周期
# for epoch in range(100):
    
#     # 调用 val_during_training 函数计算训练集的 Dice 系数
#     train_dc = val_during_training(train_dataloader)
#     # 打印训练集的平均 Dice 系数以及每个类别的平均 Dice 系数
#     print("train Dice: ", np.mean(train_dc, axis=0))
#     print("train_dice, mean, std: ", np.mean(train_dc), np.std(np.mean(train_dc, 1)))
    
#     # 验证验证集的 Dice 系数
#     val_dc = val_during_training(val_dataloader)
#     # 打印验证集的平均 Dice 系数以及每个类别的平均 Dice 系数
#     print("val Dice: ", np.mean(val_dc, axis=0))
#     print("val_dice, mean, std: ", np.mean(val_dc), np.std(np.mean(val_dc, 1)))
#     # 使用 TensorBoard 的 writer 记录训练集和验证集的平均 Dice 系数。
#     # writer.add_scalars('data/Dice', {'train': np.mean(train_dc), 'val':  np.mean(val_dc)}, epoch)    

#     # 根据验证集的平均 Dice 系数调整学习率 
#     scheduler.step(np.mean(val_dc))
#     # 打印当前学习率
#     print("learning rate = {}".format(optimizer.param_groups[0]['lr']))
    
# #    dataiter = iter(train_dataloader)
# #    data, target = dataiter.next()
    
#     # 遍历训练集的每个批次
#     for batch_idx, (data, target) in enumerate(train_dataloader):
#         data = data.squeeze()
#         target = target.squeeze()
#         # 调用 train_step 函数进行前向传播、计算损失、反向传播和参数更新
#         loss = train_step(data, target)

#         # 打印当前批次的损失
#         print("[{}:{}/{}]  LOSS={:.4}".format(epoch, 
#             batch_idx, len(train_dataloader), loss))
#         # 使用 TensorBoard 记录当前批次的损失
#         # writer.add_scalar('Train/Loss', loss, epoch*len(train_dataloader) + batch_idx)

#     # 将当前周期的训练集平均 Dice 系数存储在 train_dice 数组中
#     train_dice[epoch % 5] = np.mean(train_dc)
#     # 打印最近五个周期的训练集 Dice 系数
#     print("last five train Dice: ",train_dice)
#     # 如果最近五个周期的 Dice 系数的标准差小于等于0.00001，保存模型并结束训练
#     if np.std(np.array(train_dice)) <= 0.00001:
#         torch.save(model.state_dict(), os.path.join('trained_models', model_name+'_'+str(fold)+"_final.pkl"))
#         break
#     # 否则，每个周期结束后保存一次模型
#     torch.save(model.state_dict(), os.path.join('trained_models', model_name+'_'+str(fold)+".pkl"))