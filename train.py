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

# from sphericalunet.utils.utils import compute_weight
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
writer = SummaryWriter('log/a')

from model import Unet_40k, Unet_160k

################################################################
""" hyper-parameters """
# 指定使用第一个 GPU 进行训练。如果有多个 GPU，可以根据需要更改设备索引
cuda = torch.device('cuda:0')
# 每个批次中包含的样本数。在训练和验证过程中，将数据分割成多个批次处理
batch_size = 1
model_name = 'Unet_40k'  # 'Unet_16==40k', 'Unet_160k'
up_layer = 'upsample_interpolation' # 'upsample_interpolation', 'upsample_fixindex' 
in_channels = 3
out_channels = 36
learning_rate = 0.001
# 动量项，用于优化器中，帮助加速收敛
momentum = 0.99
# 权重衰减（L2 正则化）系数，用于防止过拟合
weight_decay = 0.0001
fold = 1 # 1,2,3 
################################################################


# class BrainSphere(torch.utils.data.Dataset):

#     # 将所有给定目录中的 .mat 文件路径收集到一个列表 self.files 中
#     def __init__(self, root1, root2 = None, root3 = None, root4 = None, root5 = None, root6 = None, root7=None):

#         # 使用 glob.glob 找到 root1 目录中所有扩展名为 .mat 的文件，并将其路径添加到 self.files 列表中
#         # 使用 sorted 对文件路径进行排序，以确保有序
#         self.files = sorted(glob.glob(os.path.join(root1, '*.mat')))  
#         # 对每一个额外的目录参数（root2 到 root7），如果不为 None，就使用 glob.glob 找到该目录中所有 .mat 文件，并将其路径添加到 self.files 列表中。
#         # 每次添加时，同样使用 sorted 对新添加的文件进行排序。  
#         if root2 is not None:
#             self.files = self.files + sorted(glob.glob(os.path.join(root2, '*.mat')))
#         if root3 is not None:
#             self.files = self.files + sorted(glob.glob(os.path.join(root3, '*.mat')))
#         if root4 is not None:
#             self.files = self.files + sorted(glob.glob(os.path.join(root4, '*.mat')))
#         if root5 is not None:
#             self.files = self.files + sorted(glob.glob(os.path.join(root5, '*.mat')))
#         if root6 is not None:
#             self.files = self.files + sorted(glob.glob(os.path.join(root6, '*.mat')))
#         if root7 is not None:
#             self.files = self.files + sorted(glob.glob(os.path.join(root7, '*.mat')))


#     def __getitem__(self, index):
#         # 根据传入的索引，从文件列表self.files中获取相应的文件路径
#         file = self.files[index]
#         # 使用sio.loadmat函数加载MAT文件中的数据。假设MAT文件中包含一个名为'data'的数组
#         data = sio.loadmat(file)
#         data = data['data']
        
#         # 从加载的数据中提取特定列（假设是第0, 1, 2列）作为特征数据。
#         feats = data[:,[0,1,2]]
#         # 计算每个特征的最大值，逐列归一化特征数据，使每个特征值在0到1之间。这样做有助于消除不同特征值范围对模型训练的影响
#         feat_max = np.max(feats,0)
#         for i in range(np.shape(feats)[1]):
#             feats[:,i] = feats[:, i]/feat_max[i]
	
#         # 加载与数据文件同名但扩展名为.label的标签文件
#         label = sio.loadmat(file[:-4] + '.label')
#         # 提取标签数组
#         label = label['label']   
#         # 通过squeeze函数去除多余的维度
#         label = np.squeeze(label)
#         # 标签减1，通常是为了将标签从1-based索引转换为0-based索引，适应某些机器学习库的要求
#         label = label - 1
#         # 返回归一化处理后的特征数据和标签，并将它们转换为指定的数据类型
#         return feats.astype(np.float32), label.astype(np.long)

#     # 返回 self.files 列表的长度，即文件的数量。
#     def __len__(self):
#         return len(self.files)

class BrainDatasetSDF(torch.utils.data.Dataset):
    def __init__(self, *data_dirs, transform=None):
        self.data_files = []
        for data_dir in data_dirs:
            self.data_files.extend(sorted(glob.glob(os.path.join(data_dir, '*_sdf.npy'))))
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = self.data_files[idx]
        data = np.load(data_path, allow_pickle=True).item()
        features = data['vertices']
        sdf = data['sdf']

        additional_features = []
        for key in ['sulc', 'curv']:
            if key in data:
                additional_features.append(data[key])
        
        if additional_features:
            additional_features = np.stack(additional_features, axis=-1)
            features = np.hstack((features, additional_features))

        if self.transform:
            features, sdf = self.transform(features, sdf)

        # return torch.tensor(features, dtype=torch.float32), torch.tensor(sdf, dtype=torch.float32)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(sdf, dtype=torch.long)


# 定义数据集目录路径
fold1 = '/mnt/d/Spherical_U-Net/Test/lh/fold1'
fold2 = '/mnt/d/Spherical_U-Net/Test/lh/fold2'
fold3 = '/mnt/d/Spherical_U-Net/Test/lh/fold3'
fold4 = '/mnt/d/Spherical_U-Net/Test/lh/fold4'
fold5 = '/mnt/d/Spherical_U-Net/Test/lh/fold5'
fold6 = '/mnt/d/Spherical_U-Net/Test/lh/fold6'

if fold == 1:
    train_dataset = BrainDatasetSDF(fold3, fold6, fold2, fold5)          
    val_dataset = BrainDatasetSDF(fold1)
elif fold == 2:
    train_dataset = BrainDatasetSDF(fold1, fold4, fold3, fold6)          
    val_dataset = BrainDatasetSDF(fold2)
elif fold == 3:
    train_dataset = BrainDatasetSDF(fold1, fold4, fold2, fold5)          
    val_dataset = BrainDatasetSDF(fold3)
else:
    raise NotImplementedError('fold name is wrong!')

# batch_size：每个批次的样本数，通常与训练集相同
# shuffle=True：在每个 epoch 开始时打乱数据，这样可以增加数据的随机性，防止模型过拟合。
# pin_memory=True：如果设置为 True，DataLoader 会在返回之前将 Tensors 的数据复制到 CUDA 固定内存中，可以加速主机到 GPU 的数据传输。
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)



# if model_name == 'Unet_infant':
#     model = Unet_infant(in_ch=in_channels, out_ch=out_channels)
# elif model_name == 'Unet_18':
#      model = Unet_18(in_ch=in_channels, out_ch=out_channels)
# elif model_name == 'Unet_2ring':
#     model = Unet_2ring(in_ch=in_channels, out_ch=out_channels)
# elif model_name == 'Unet_repa':
#     model = Unet_repa(in_ch=in_channels, out_ch=out_channels)
# elif model_name == 'fcn':
#     model = fcn(in_ch=in_channels, out_ch=out_channels)
# elif model_name == 'SegNet':
#     model = SegNet(in_ch=in_channels, out_ch=out_channels, up_layer=up_layer)
# elif model_name == 'SegNet_max':
#     model = SegNet_max(in_ch=in_channels, out_ch=out_channels)
if model_name == 'Unet_40k':
    model = Unet_40k(in_ch=in_channels, out_ch=out_channels)
elif model_name == 'Unet_160k':
    model = Unet_160k(in_ch=in_channels, out_ch=out_channels)
else:
    raise NotImplementedError('model name is wrong!')

# model.parameters()：返回模型的所有参数。
# x.numel()：返回参数张量 x 中的元素总数。
# sum(x.numel() for x in model.parameters())：计算模型所有参数的元素总数，即模型的总参数数量。
print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
# 将模型移动到指定的 GPU 上进行计算
model.cuda(cuda)
# 使用交叉熵损失函数 nn.CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
# 使用 Adam 优化器 torch.optim.Adam 来优化模型参数
# lr=learning_rate：设置学习率
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 使用学习率调度器 torch.optim.lr_scheduler.ReduceLROnPlateau
# optimizer：与之关联的优化器。
# 'max'：监控的指标，如果指标没有改善，学习率将减少。
# factor=0.2：当指标停止改善时，将学习率减少到原来的 20%。
# patience=1：如果指标没有改善超过 patience 个周期，则减少学习率。
# verbose=True：启用详细输出。
# threshold=0.0001：用于确定何时降低学习率的阈值。
# threshold_mode='rel'：使用相对变化来确定何时降低学习率。
# min_lr=0.000001：设置学习率的下限。
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=0.000001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=1, threshold=0.0001, threshold_mode='rel', min_lr=0.000001)


def train_step(data, target):
    # 设置模型为训练模式
    model.train()
    # 将数据和标签移动到GPU
    data, target = data.cuda(cuda).float(), target.cuda(cuda).long()

    # 前向传播
    prediction = model(data)
    # 计算损失
    loss = criterion(prediction, target)
    # 清零梯度
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新模型参数
    optimizer.step()
    # 返回损失值
    return loss.item()


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

def compute_dice(pred, gt):
    # 使用 .cpu().numpy() 方法将预测结果 pred 和真实标签 gt 
    # 从 GPU 内存移动到 CPU 内存，并转换为 NumPy 数组。
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    
    # 创建一个长度为 36 的零数组，用于存储每个类别的 Dice 系数。假设有 36 个类别。
    dice = np.zeros(36)
    for i in range(36):
        # 使用 np.where(gt == i)[0] 找到真实标签中类别 i 的索引。
        gt_indices = np.where(gt == i)[0]
        # 使用 np.where(pred == i)[0] 找到预测结果中类别 i 的索引。
        pred_indices = np.where(pred == i)[0]
        
        if len(gt_indices) == 0 or len(pred_indices) == 0:
            # 跳过该类别的计算
            dice[i] = 1.0 if len(gt_indices) == 0 and len(pred_indices) == 0 else 0.0
        else:
            # 使用 np.intersect1d(gt_indices, pred_indices) 找到真实标签和预测结果中类别 i 的交集索引。
            # 计算类别 i 的 Dice 系数
            dice[i] = 2 * len(np.intersect1d(gt_indices, pred_indices)) / (len(gt_indices) + len(pred_indices))
    return dice


def val_during_training(dataloader):
    # 将模型设置为评估模式。这会禁用 dropout 层和 batch normalization 层的训练行为
    model.eval()

    # 创建一个零数组 dice_all，用于存储每个批次和每个类别的 Dice 系数。假设有 36 个类别，len(dataloader) 是验证数据集的批次数。
    dice_all = np.zeros((len(dataloader),36))
    # 使用 enumerate 遍历 dataloader 中的每个批次
    for batch_idx, (data, target) in enumerate(dataloader):
        # data.squeeze() 和 target.squeeze()：移除维度为 1 的维度。
        data = data.squeeze().float()
        target = target.squeeze().long()
        # 将数据和标签移动到 GPU 上
        data, target = data.cuda(cuda), target.cuda(cuda)
        # with torch.no_grad()：在上下文管理器 torch.no_grad() 中进行前向传播，禁用梯度计算，以减少内存使用和加速计算。
        with torch.no_grad():
            prediction = model(data)
        # 使用 prediction.max(1)[1] 找到预测结果中每个像素的最大值索引，即预测的类别。    
        prediction = prediction.max(1)[1]
        # 计算当前批次的 Dice 系数，并存储在 dice_all 数组中。
        dice_all[batch_idx,:] = compute_dice(prediction, target)

    return dice_all


train_dice = [0, 0, 0, 0, 0]
# 循环100个训练周期
for epoch in range(100):
    
    # 调用 val_during_training 函数计算训练集的 Dice 系数
    train_dc = val_during_training(train_dataloader)
    # 打印训练集的平均 Dice 系数以及每个类别的平均 Dice 系数
    print("train Dice: ", np.mean(train_dc, axis=0))
    print("train_dice, mean, std: ", np.mean(train_dc), np.std(np.mean(train_dc, 1)))
    
    # 验证验证集的 Dice 系数
    val_dc = val_during_training(val_dataloader)
    # 打印验证集的平均 Dice 系数以及每个类别的平均 Dice 系数
    print("val Dice: ", np.mean(val_dc, axis=0))
    print("val_dice, mean, std: ", np.mean(val_dc), np.std(np.mean(val_dc, 1)))
    # 使用 TensorBoard 的 writer 记录训练集和验证集的平均 Dice 系数。
    writer.add_scalars('data/Dice', {'train': np.mean(train_dc), 'val':  np.mean(val_dc)}, epoch)    

    # 根据验证集的平均 Dice 系数调整学习率 
    scheduler.step(np.mean(val_dc))
    # 打印当前学习率
    print("learning rate = {}".format(optimizer.param_groups[0]['lr']))
    
#    dataiter = iter(train_dataloader)
#    data, target = dataiter.next()
    
    # 遍历训练集的每个批次
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.squeeze()
        target = target.squeeze()
        # 调用 train_step 函数进行前向传播、计算损失、反向传播和参数更新
        loss = train_step(data, target)

        # 打印当前批次的损失
        print("[{}:{}/{}]  LOSS={:.4}".format(epoch, 
              batch_idx, len(train_dataloader), loss))
        # 使用 TensorBoard 记录当前批次的损失
        writer.add_scalar('Train/Loss', loss, epoch*len(train_dataloader) + batch_idx)

    # 将当前周期的训练集平均 Dice 系数存储在 train_dice 数组中
    train_dice[epoch % 5] = np.mean(train_dc)
    # 打印最近五个周期的训练集 Dice 系数
    print("last five train Dice: ",train_dice)
    # 如果最近五个周期的 Dice 系数的标准差小于等于0.00001，保存模型并结束训练
    if np.std(np.array(train_dice)) <= 0.00001:
        torch.save(model.state_dict(), os.path.join('trained_models_cc', model_name+'_'+str(fold)+"_final.pkl"))
        break
    # 否则，每个周期结束后保存一次模型
    torch.save(model.state_dict(), os.path.join('trained_models_cc', model_name+'_'+str(fold)+".pkl"))