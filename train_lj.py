# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import torch
# import torch.nn as nn
# import torchvision
# import scipy.io as sio 
# import numpy as np
# import glob
# import os
# import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter # type: ignore
# writer = SummaryWriter('log/a')

# from model import Unet_40k, Unet_160k

# ################################################################
# """ hyper-parameters """
# cuda = torch.device('cuda:0')
# batch_size = 1
# fold = 1 # 1,2,3 
# model_name = 'Unet_40k'  # 'Unet_40k', 'Unet_160k'
# up_layer = 'upsample_interpolation' # 'upsample_interpolation', 'upsample_fixindex' 
# in_channels = 2
# out_channels = 1 ## by Jiale
# learning_rate = 0.001
# momentum = 0.99
# # 权重衰减（L2 正则化）系数，用于防止过拟合
# weight_decay = 0.0001

# torch.cuda.empty_cache()

# ################################################################

# class BrainSphere(torch.utils.data.Dataset):
#     def __init__(self, *data_dirs):
#         self.data_files = []
#         for data_dir in data_dirs:
#             files = sorted(glob.glob(os.path.join(data_dir, '*_sdf.npz')))
#             self.data_files.extend(files) 

#     # def __getitem__(self, index):
#     #     file = self.data_files[index]
#     #     # data = np.load(file, allow_pickle=True).item()
#     #     data = np.load(file, allow_pickle=True)
        
#     #     # Extract features
#     #     sulc = data['sulc']
#     #     curv = data['curv']
#     #     feats = np.stack((sulc, curv), axis=1)
        
#     #     # Normalize features
#     #     feat_max = np.max(feats, axis=0)
#     #     for i in range(feats.shape[1]):
#     #         feats[:, i] = feats[:, i] / feat_max[i]
        
#     #     # Extract labels
#     #     label = data['sdf']
#     #     label = np.squeeze(label) - 1

#     #     label = label.reshape(-1, 1)
        
#     #     return torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

#     def __getitem__(self, index):
#         file = self.data_files[index]
#         data = np.load(file, allow_pickle=True)
        
#         sulc = data['sulc']
#         curv = data['curv']
#         feats = np.stack((sulc, curv), axis=1)
        
#         # Normalize each feature independently
#         feat_max = np.max(feats, axis=0, keepdims=True)
#         feats = feats / feat_max
        
#         label = data['sdf']
#         label = np.squeeze(label) - 1
#         label = label.reshape(-1, 1)
        
#         return torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


#     def __len__(self):
#         return len(self.data_files)

# fold1 = '/home/lab/Documents/Spherical_U-Net/Test1/lh/fold1'
# fold2 = '/home/lab/Documents/Spherical_U-Net/Test1/lh/fold2'
# fold3 = '/home/lab/Documents/Spherical_U-Net/Test1/lh/fold3'
# fold4 = '/home/lab/Documents/Spherical_U-Net/Test1/lh/fold4'
# fold5 = '/home/lab/Documents/Spherical_U-Net/Test1/lh/fold5'
# fold6 = '/home/lab/Documents/Spherical_U-Net/Test1/lh/fold6'

# if fold == 1:
#     train_dataset = BrainSphere(fold3,fold6,fold2,fold5)
#     val_dataset = BrainSphere(fold1)
# elif fold == 2:
#     train_dataset = BrainSphere(fold1,fold4,fold3,fold6)
#     val_dataset = BrainSphere(fold2)
# elif fold == 3:
#     train_dataset = BrainSphere(fold1,fold4,fold2,fold5)
#     val_dataset = BrainSphere(fold3)
# else:
#     raise NotImplementedError('fold name is wrong!')

# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


# ##########################################################################################################

# if model_name == 'Unet_40k':
#     model = Unet_40k(in_ch=in_channels, out_ch=out_channels)
# elif model_name == 'Unet_160k':
#     model = Unet_160k(in_ch=in_channels, out_ch=out_channels)
# else:
#     raise NotImplementedError('model name is wrong!')

# print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
# model.cuda(cuda)

# criterion = nn.L1Loss()

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=0.000001)


# def train_step(data, target):
#     model.train()

#     # data = data.squeeze()
#     # target = target.squeeze()
#     data, target = data.cuda(cuda), target.cuda(cuda)

#     prediction = model(data)

#     target = target.view_as(prediction) # Fix 1

#     loss = criterion(prediction, target)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     return loss.item()

# def compute_mae(pred, gt):
#     # pred = pred.cpu().numpy()
#     # gt = gt.cpu().numpy()   
#     pred = pred.cpu().numpy().flatten()
#     gt = gt.cpu().numpy().flatten()

#     mae = np.mean(np.abs(pred - gt))
#     return mae


# def val_during_training(dataloader):
#     model.eval()

#     # mae_all = np.zeros((len(dataloader),36))
#     mae_all = []

#     for batch_idx, (data, target) in enumerate(dataloader):
#         # # data.squeeze() 和 target.squeeze()：移除维度为 1 的维度。
#         data = data.squeeze()
#         target = target.squeeze()
#         data, target = data.cuda(cuda), target.cuda(cuda)
#         with torch.no_grad():
#             prediction = model(data)

#         # # 使用 prediction.max(1)[1] 找到预测结果中每个像素的最大值索引，即预测的类别。    
#         # prediction = prediction.max(1)[1]

#         # target = target.view_as(prediction) # Fix 1

#         prediction = prediction.view_as(target) 

#         # mae_all[batch_idx,:] = compute_mae(prediction, target) # By Jiale
#         mae_all.append(compute_mae(prediction, target))

#     # return dice_all
#     return np.array(mae_all)


# train_mae = [0, 0, 0, 0, 0]
# for epoch in range(100):
    
#     train_dc = val_during_training(train_dataloader)

#     print("train mae: ", np.mean(train_dc, axis=0))
#     # print("train_mae, mean, std: ", np.mean(train_dc), np.std(np.mean(train_dc, 1)))

#     train_mean = np.mean(train_dc, axis=0)
#     train_std = np.std(train_mean)
#     print("train_mae, mean, std:", np.mean(train_dc), train_std)


#     val_dc = val_during_training(val_dataloader)

#     # print("val mae: ", np.mean(val_dc, axis=0))
#     # # print("val_mae, mean, std: ", np.mean(val_dc), np.std(np.mean(val_dc, 1)))
#     # print("val_mae, mean, std: ", np.mean(val_dc), np.std(np.mean(val_dc)))
#     val_mean = np.mean(val_dc)
#     val_std = np.std(val_dc)

#     print("val_mae, mean, std:", val_mean, val_std)
#     # writer.add_scalars('data/Dice', {'train': np.mean(train_dc), 'val':  np.mean(val_dc)}, epoch)    
#     writer.add_scalars('data/mae', {'train': np.mean(train_dc), 'val':  np.mean(val_dc)}, epoch)    


#     scheduler.step(np.mean(val_dc))
#     print("learning rate = {}".format(optimizer.param_groups[0]['lr']))
    
# #    dataiter = iter(train_dataloader)
# #    data, target = dataiter.next()
    
#     for batch_idx, (data, target) in enumerate(train_dataloader):
#         data = data.squeeze()
#         target = target.squeeze()
#         loss = train_step(data, target)

#         print("[{}:{}/{}]  LOSS={:.4}".format(epoch, 
#             batch_idx, len(train_dataloader), loss))
#         # writer.add_scalar('Train/Loss', loss, epoch*len(train_dataloader) + batch_idx)
#         writer.add_scalar('Train/Loss', loss, epoch*len(train_dataloader) + batch_idx)


#     train_mae[epoch % 5] = np.mean(train_dc)
#     print("last five train mae: ",train_mae)
#     torch.save(model.state_dict(), os.path.join('trained_models', model_name+'_'+str(fold)+".pkl"))




################################################################
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
batch_size = 1
fold = 3  # 1,2,3
model_name = 'Unet_40k'  # 'Unet_40k', 'Unet_160k'
up_layer = 'upsample_interpolation'  # 'upsample_interpolation', 'upsample_fixindex'
in_channels = 2
out_channels = 1  # by Jiale
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
            files = sorted(glob.glob(os.path.join(data_dir, '*_sdf.npz')))
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

        # 提取标签
        sdf = data['sdf']
        sdf = np.squeeze(sdf)

        return torch.tensor(feats, dtype=torch.float32), torch.tensor(sdf, dtype=torch.float32)

    def __len__(self):
        return len(self.data_files)

fold1 = './Test1/lh/fold1'
fold2 = './Test1/lh/fold2'
fold3 = './Test1/lh/fold3'
fold4 = './Test1/lh/fold4'
fold5 = './Test1/lh/fold5'
fold6 = './Test1/lh/fold6'

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

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=0.000001)

def train_step(data, target):
    model.train()
    data, target = data.cuda(cuda), target.cuda(cuda)
    prediction = model(data)
    target = target.view_as(prediction)  # 确保形状一致
    loss = criterion(prediction, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def compute_mae(pred, gt):
    pred = pred.cpu().numpy().flatten()
    gt = gt.cpu().numpy().flatten()
    mae = np.mean(np.abs(pred - gt))
    return mae

def val_during_training(dataloader):
    model.eval()
    mae_all = []
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.squeeze()
        target = target.squeeze()
        data, target = data.cuda(cuda), target.cuda(cuda)
        with torch.no_grad():
            prediction = model(data)
        prediction = prediction.view_as(target)
        mae_all.append(compute_mae(prediction, target))
    return np.array(mae_all)

train_mae = [0, 0, 0, 0, 0]
for epoch in range(100):
    train_dc = val_during_training(train_dataloader)
    train_mean = np.mean(train_dc, axis=0)
    train_std = np.std(train_mean)
    print("train_mae, mean, std:", np.mean(train_dc), train_std)
    
    val_dc = val_during_training(val_dataloader)
    val_mean = np.mean(val_dc)
    val_std = np.std(val_mean)
    print("val_mae, mean, std:", val_mean, val_std)
    writer.add_scalars('data/mae', {'train': np.mean(train_dc), 'val': np.mean(val_dc)}, epoch)
    
    scheduler.step(np.mean(val_dc))
    print("learning rate = {}".format(optimizer.param_groups[0]['lr']))
    
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.squeeze()
        target = target.squeeze()
        loss = train_step(data, target)
        print("[{}:{}/{}]  LOSS={:.4}".format(epoch, batch_idx, len(train_dataloader), loss))
        writer.add_scalar('Train/Loss', loss, epoch * len(train_dataloader) + batch_idx)
    
    train_mae[epoch % 5] = np.mean(train_dc)
    print("last five train mae:", train_mae)
    torch.save(model.state_dict(), os.path.join('trained_models', model_name + '_' + str(fold) + ".pkl"))
