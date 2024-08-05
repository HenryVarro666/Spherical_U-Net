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

from model import Unet_40k, Unet_160k, Unet_40k_batch, Unet_40k_batch_2

################################################################
""" hyper-parameters """
cuda = torch.device('cuda:0')
batch_size = 4
fold = 1  # 1,2,3
model_name = 'Unet_40k_batch_2'  # 'Unet_40k', 'Unet_160k'
up_layer = 'upsample_interpolation'  # 'upsample_interpolation', 'upsample_fixindex'
in_channels = 2
# out_channels = 2  # by Jiale
out_ch1 = 1
out_ch2 = 1
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

        sulc = data['sulc']
        curv = data['curv']
        feats = np.stack((sulc, curv), axis=1)

        feat_max = np.max(feats, axis=0, keepdims=True)
        feats = feats / feat_max
        # print(feats.shape)

        # 提取标签
        ring_distances = data['ring_distances']
        label_reg = np.squeeze(ring_distances)

        line_mask = data['line_mask']
        label_cla = np.squeeze(line_mask)


        # print(label.shape)
        # label = np.expand_dims(label, axis=0)  # Add a channel dimension if necessary

        return torch.tensor(feats, dtype=torch.float32), torch.tensor(label_reg, dtype=torch.float32), torch.tensor(label_cla, dtype=torch.float32)

    def __len__(self):
        return len(self.data_files)

fold1 = './Test_cla_reg/lh/fold1'
fold2 = './Test_cla_reg/lh/fold2'
fold3 = './Test_cla_reg/lh/fold3'
fold4 = './Test_cla_reg/lh/fold4'
fold5 = './Test_cla_reg/lh/fold5'
fold6 = './Test_cla_reg/lh/fold6'


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

if model_name == 'Unet_40k_batch_2':
    # model = Unet_40k_batch(in_ch=in_channels, out_ch=out_channels)
    model = Unet_40k_batch_2(in_ch=in_channels, out_ch1=out_ch1, out_ch2=out_ch2)

elif model_name == 'Unet_160k':
    model = Unet_160k(in_ch=in_channels, out_ch=out_ch1)
else:
    raise NotImplementedError('model name is wrong!')

print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda(cuda)

# weight = np.array([426, 413]).astype(np.float32) / (426+413)


criterion_reg = nn.MSELoss()
criterion_cla = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()
# criterion_reg = nn.MSELoss()

# criterion = nn.FocalLoss(gamma=1.0, alpha=weight)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=0.000001)

def train_step(data, target_reg, target_cla):
    model.train()
    data, target_reg, target_cla = data.cuda(cuda), target_reg.cuda(cuda), target_cla.cuda(cuda)
    prediction_reg, prediction_cla = model(data)
    target_reg = target_reg.view_as(prediction_reg)  # 确保形状一致
    target_cla = target_cla.view_as(prediction_cla)  # 确保形状一致

    # hyper: weight for regression loss
    hyper = 0.1
    loss = criterion_cla(prediction_cla, target_cla) + hyper * criterion_reg(prediction_reg, target_reg)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def compute_dice(pred, gt):
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

    # 用于存储每个批次的 Dice 系数和 MAE
    dice_all = []
    mae_all = []

    # 使用 enumerate 遍历 dataloader 中的每个批次
    for batch_idx, (data, target_reg, target_cla) in enumerate(dataloader):

        # 将数据和标签移动到 GPU 上
        data, target_reg, target_cla = data.cuda(cuda), target_reg.cuda(cuda), target_cla.cuda(cuda)

        # data.squeeze() 和 target.squeeze()：移除维度为 1 的维度。
        data = data.squeeze()
        target_reg = target_reg.squeeze()
        target_cla = target_cla.squeeze()
        
        # with torch.no_grad()：在上下文管理器 torch.no_grad() 中进行前向传播，禁用梯度计算，以减少内存使用和加速计算。
        with torch.no_grad():
            prediction_reg, prediction_cla = model(data)

        # 调整 target_reg 的形状以匹配 prediction_reg
        if prediction_reg.shape != target_reg.shape:
            target_reg = target_reg.view_as(prediction_reg)
        
        # 使用 prediction.max(1)[1] 找到预测结果中每个像素的最大值索引，即预测的类别。    
        prediction_cla = prediction_cla.max(1)[1]

        # 计算当前批次的 Dice 系数，并存储在 dice_all 列表中。
        dice = compute_dice(prediction_cla, target_cla)
        dice_all.append(dice)

        # 计算当前批次的 MAE，并存储在 mae_all 列表中
        mae = torch.mean(torch.abs(prediction_reg - target_reg)).item()
        mae_all.append(mae)

    # 将列表转换为 numpy 数组
    dice_all = np.array(dice_all)
    mae_all = np.array(mae_all)

    return dice_all, mae_all

# 调用 val_during_training 函数时处理返回的两个评估结果
def evaluate_model(dataloader):
    dice_all, mae_all = val_during_training(dataloader)

    # 打印 Dice 系数的结果
    print("Dice Coefficients: ", np.mean(dice_all, axis=0))
    print("Mean Dice Coefficient: ", np.mean(dice_all))
    print("Standard Deviation of Dice Coefficient: ", np.std(np.mean(dice_all, 1)))

    # 打印 MAE 的结果
    print("Mean Absolute Error (MAE): ", np.mean(mae_all))
    print("Standard Deviation of MAE: ", np.std(mae_all))


train_dice = [0, 0, 0, 0, 0]
min_epochs = 5  # 最少训练的周期数
for epoch in range(100):
    
    train_dc, train_mae = val_during_training(train_dataloader)
    print("train Dice: ", np.mean(train_dc, axis=0))
    print("train_dice, mean, std: ", np.mean(train_dc), np.std(np.mean(train_dc, axis=0)))
    print("train MAE: ", np.mean(train_mae), "MAE std: ", np.std(train_mae))
    
    val_dc, val_mae = val_during_training(val_dataloader)
    print("val Dice: ", np.mean(val_dc, axis=0))
    print("val_dice, mean, std: ", np.mean(val_dc), np.std(np.mean(val_dc, axis=0)))
    print("val MAE: ", np.mean(val_mae), "MAE std: ", np.std(val_mae))
    writer.add_scalars('data/Dice', {'train': np.mean(train_dc), 'val':  np.mean(val_dc)}, epoch)    
    writer.add_scalars('data/MAE', {'train': np.mean(train_mae), 'val': np.mean(val_mae)}, epoch)
    
    scheduler.step(np.mean(val_dc))
    print("learning rate = {}".format(optimizer.param_groups[0]['lr']))
    
    for batch_idx, (data, target_reg, target_cla) in enumerate(train_dataloader):
        # data.squeeze() 和 target.squeeze()：移除维度为 1 的维度。
        data = data.squeeze()
        target_reg = target_reg.squeeze()
        target_cla = target_cla.squeeze()
        data, target_reg, target_cla = data.cuda(cuda), target_reg.cuda(cuda), target_cla.cuda(cuda)
        loss = train_step(data, target_cla, target_reg)

        # 打印当前批次的损失
        print("[{}:{}/{}]  LOSS={:.4}".format(epoch, 
              batch_idx, len(train_dataloader), loss))
        # 使用 TensorBoard 记录当前批次的损失
        writer.add_scalar('Train/Loss', loss, epoch*len(train_dataloader) + batch_idx)

    # 将当前周期的训练集平均 Dice 系数存储在 train_dice 数组中
    train_dice[epoch % 5] = np.mean(train_dc)
    # 打印最近五个周期的训练集 Dice 系数
    print("last five train Dice: ", train_dice)

    # Define the output directory
    output_dir = 'trained_models_cla_reg'
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 如果最近五个周期的 Dice 系数的标准差小于等于0.00001，保存模型并结束训练
    if epoch >= min_epochs and np.std(np.array(train_dice)) <= 0.00001:
        print(f"Training stopped early at epoch {epoch} due to low standard deviation in Dice scores.")
        torch.save(model.state_dict(), os.path.join(output_dir, model_name+'_'+str(fold)+"_final.pkl"))
        break
    # 否则，每个周期结束后保存一次模型
    torch.save(model.state_dict(), os.path.join(output_dir, model_name+'_'+str(fold)+".pkl"))

