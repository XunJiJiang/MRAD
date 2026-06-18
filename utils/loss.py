import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

# ===================== FocalLoss: 带标签平滑的Focal Loss实现（用于分割任务） =====================
class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        # 校验标签平滑值是否在合法范围 [0, 1] 内
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        # 如果指定了非线性激活函数，先对 logit 应用
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        # 将多维 logit 展平为 (N, C) 格式，便于逐像素计算 loss
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        # 将 target 展平为 (N, 1) 格式
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        # 根据 alpha 的类型构造类别权重张量
        if alpha is None:
            # alpha 为空时，各类别权重均为 1
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            # alpha 为列表或数组时，检查长度与类别数一致，并归一化
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            # alpha 为单个浮点数时，用于指定 balance_index 对应类别的权重
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        # 将 alpha 移动到与 logit 相同的设备
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        # 构造 one-hot 编码的目标张量
        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        # 若启用标签平滑，将 one-hot 值限制在 [smooth/(C-1), 1-smooth] 区间内
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        # 计算预测概率 pt 和对数概率 logpt
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        # 获取当前 target 对应的 alpha 权重
        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        # 计算 Focal Loss: -alpha * (1-pt)^gamma * log(pt)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        # 按 size_average 决定返回均值还是逐元素 loss
        if self.size_average:
            loss = loss.mean()
        return loss


# ===================== BinaryDiceLoss: 二值 Dice Loss（用于分割任务） =====================
class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        # 计算 Dice 系数（带平滑项避免除零）
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss

# ===================== smooth: 平滑正则化损失 =====================
# 计算数组相邻元素之间的差异平方和，用于鼓励输出平滑性
def smooth(arr, lamda1):
    new_array = arr
    arr2 = torch.zeros_like(arr)
    # 沿第1维（行方向）计算相邻差分：arr2[i] = arr[i+1]，最后一行保持不变
    arr2[:, :-1, :] = arr[:, 1:, :]
    arr2[:, -1, :] = arr[:, -1, :]

    new_array2 = torch.zeros_like(new_array)
    # 沿第2维（列方向）计算相邻差分：new_array2[i] = new_array[i+1]，最后一列保持不变
    new_array2[:, :, :-1] = new_array[:, :, 1:]
    new_array2[:, :, -1] = new_array[:, :, -1]
    # 对行方向和列方向的差分平方求和后平均
    loss = (torch.sum((arr2 - arr) ** 2) + torch.sum((new_array2 - new_array) ** 2)) / 2
    return lamda1 * loss

# ===================== sparsity: 稀疏性正则化损失 =====================
# 鼓励输出趋向全0或全1，target=0 时惩罚非零值，target!=0 时惩罚非一值
def sparsity(arr, target, lamda2):
    if target == 0:
        # 计算每列 L2 范数的均值，鼓励输出接近 0
        loss = torch.mean(torch.norm(arr, dim=0))
    else:
        # 计算每列 (1-arr) 的 L2 范数的均值，鼓励输出接近 1
        loss = torch.mean(torch.norm(1-arr, dim=0))
    return lamda2 * loss
