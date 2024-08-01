#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import scipy.io as sio
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter  # type: ignore

writer = SummaryWriter('log/a')

from model import Unet_40k, Unet_160k

################################################################
""" hyper-parameters """
cuda = torch.device('cuda:0')
batch_size = 4
fold = 1  # 1,2,3
model_name = 'Unet_40k'  # 'Unet_40k', 'Unet_160k'
up_layer = 'upsample_interpolation'  # 'upsample_interpolation', 'upsample_fixindex'
in_channels = 2
out_channels = 3  # by Jiale
learning_rate = 0.001
momentum = 0.99
# 权重衰减（L2 正则化）系数，用于防止过拟合
weight_decay = 0.0001

torch.cuda.empty_cache()

################################################################

class BrainSphere(torch.utils.data.Dataset):
    def __init__(self, *data_dirs):
        self.data_files = []
        for data_dir in data_dirs:
            files = sorted(glob.glob(os.path.join(data_dir, '*_linemask_skeleton.npz')))
            self.data_files.extend(files)

    def __getitem__(self, index):
        file = self.data_files[index]
        data = np.load(file, allow_pickle=True)

        # 提取特征
        sulc = data['sulc']
        curv = data['curv']
        feats = np.stack((sulc, curv), axis=1)

        # 对每个特征独立归一化
        feat_max = np.max(feats, axis=0, keepdims=True)
        feats = feats / feat_max

        # # line_mask = data['line_mask']
        # # line_mask = np.squeeze(line_mask)

        # # line_data_mask = data['line_data_mask']
        # # line_data_mask = np.squeeze(line_data_mask)
        # line_original_mask = data['line_original_mask']
        # line_mask = data['line_mask']
        # non_line_data = data['non_line_data']
        # label = np.stack((line_original_mask, line_mask, non_line_data), axis=0)
        # label = np.squeeze(label)
        # label = label - 1

        # 提取标签
        one_hot_labels = data['line_original_mask']
        label = np.squeeze(one_hot_labels)

        # label = np.expand_dims(label, axis=0)  # Add a channel dimension if necessary

        return torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.data_files)

# fold1 = './Test4/lh/fold1'
# fold2 = './Test4/lh/fold2'
# fold3 = './Test4/lh/fold3'
# fold4 = './Test4/lh/fold4'
# fold5 = './Test4/lh/fold5'
# fold6 = './Test4/lh/fold6'

fold1 = './Test3/lh/fold1'
fold2 = './Test3/lh/fold2'
fold3 = './Test3/lh/fold3'
fold4 = './Test3/lh/fold4'
fold5 = './Test3/lh/fold5'
fold6 = './Test3/lh/fold6'


if fold == 1:
    train_dataset = BrainSphere(fold3, fold6, fold2, fold5)
    val_dataset = BrainSphere(fold1)
elif fold == 2:
    train_dataset = BrainSphere(fold1, fold4, fold3, fold6)
    val_dataset = BrainSphere(fold2)
elif fold == 3:
    train_dataset = BrainSphere(fold1, fold4, fold2, fold5)
    val_dataset = BrainSphere(fold3)
else:
    raise NotImplementedError('fold name is wrong!')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

##########################################################################################################

if model_name == 'Unet_40k':
    model = Unet_40k(in_ch=in_channels, out_ch=out_channels)
elif model_name == 'Unet_160k':
    model = Unet_160k(in_ch=in_channels, out_ch=out_channels)
else:
    raise NotImplementedError('model name is wrong!')

print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda(cuda)

# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=0.000001)

def train_step(data, target):
    model.train()
    data, target = data.cuda(cuda), target.cuda(cuda)
    prediction = model(data)
    target = target.view_as(prediction)  # 确保形状一致
    # target = target.squeeze(1)  # Remove the channel dimension if it exists

    loss = criterion(prediction, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def compute_dice(pred, gt):
    # 使用 .cpu().numpy() 方法将预测结果 pred 和真实标签 gt 
    # 从 GPU 内存移动到 CPU 内存，并转换为 NumPy 数组。
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    
    dice = np.zeros(3)
    for i in range(3):
        # 使用 np.where(gt == i)[0] 找到真实标签中类别 i 的索引。
        gt_indices = np.where(gt == i)[0]
        # 使用 np.where(pred == i)[0] 找到预测结果中类别 i 的索引。
        pred_indices = np.where(pred == i)[0]
        # 使用 np.intersect1d(gt_indices, pred_indices) 找到真实标签和预测结果中类别 i 的交集索引。
        # 计算类别 i 的 Dice 系数
        dice[i] = 2 * len(np.intersect1d(gt_indices, pred_indices))/(len(gt_indices) + len(pred_indices))
    return dice

def val_during_training(dataloader):
    # 将模型设置为评估模式。这会禁用 dropout 层和 batch normalization 层的训练行为
    model.eval()

    # 创建一个零数组 dice_all，用于存储每个批次和每个类别的 Dice 系数。假设有 36 个类别，len(dataloader) 是验证数据集的批次数。
    dice_all = np.zeros((len(dataloader),3))
    # 使用 enumerate 遍历 dataloader 中的每个批次
    for batch_idx, (data, target) in enumerate(dataloader):
        # data.squeeze() 和 target.squeeze()：移除维度为 1 的维度。
        data = data.squeeze()
        target = target.squeeze()
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

    # Define the output directory
    output_dir = 'trained_models_3'
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # # 如果最近五个周期的 Dice 系数的标准差小于等于0.00001，保存模型并结束训练
    # if np.std(np.array(train_dice)) <= 0.00001:
    #     torch.save(model.state_dict(), os.path.join(output_dir, model_name+'_'+str(fold)+"_final.pkl"))
    #     break
    # 否则，每个周期结束后保存一次模型
    torch.save(model.state_dict(), os.path.join(output_dir, model_name+'_'+str(fold)+".pkl"))