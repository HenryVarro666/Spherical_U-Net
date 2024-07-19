<!--
 * @Author: HenryVarro666 1504517223@qq.com
 * @Date: 2024-07-19 10:07:53
 * @LastEditors: HenryVarro666 1504517223@qq.com
 * @LastEditTime: 2024-07-19 10:20:40
 * @FilePath: /Spherical_U-Net/Error.md
-->

## 1
/home/cxc0366/fsl/lib/python3.11/site-packages/torch/nn/modules/loss.py:101: UserWarning: Using a target size (torch.Size([40962])) that is different to the input size (torch.Size([40962, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.l1_loss(input, target, reduction=self.reduction)



## 2
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 87.51 GiB (GPU 0; 47.50 GiB total capacity; 29.21 MiB already allocated; 46.95 GiB free; 46.00 MiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

