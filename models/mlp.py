import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F
from collections import OrderedDict

# QuickGELU: GELU激活函数的近似实现
class QuickGELU(nn.Module):
    # 前向传播：x * sigmoid(1.702 * x)，近似标准GELU
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# AnomalyMLP: 将patch嵌入映射为偏置，输出正常/异常两个角度的偏置向量
# 输入两个patch嵌入，各自通过独立的线性层生成偏置
class AnomalyMLP(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        # 第一个线性层：处理第一个patch嵌入
        self.mlp1 = nn.Linear(d_model, d_model)
        # 第二个线性层：处理第二个patch嵌入
        self.mlp2 = nn.Linear(d_model, d_model)

    # 前向传播：将两个patch嵌入分别映射为偏置，堆叠后输出 [B, 2, D]
    def forward(self, patch_emb1,patch_emb2): 
        # 预测目标角度
        return torch.stack([self.mlp1(patch_emb1), self.mlp2(patch_emb2)], dim=1)

# （已注释）AnomalyMLP的另一种实现：使用MLP-Mixer风格（多层感知机）的流形学习网络
# class AnomalyMLP(nn.Module):
#     def __init__(self, d_model=768):
#         super().__init__()
#         # 流形学习网络
#         self.mlp1 = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.mlp2 = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),  
#             ("c_proj", nn.Linear(d_model * 4, d_model))]))

#     def forward(self, patch_emb): 
#         # 预测目标角度
#         return torch.stack([self.mlp1(patch_emb), self.mlp2(patch_emb)], dim=1)

# MLP: 双线性投影模块，用于图像特征的投影
# 对两个嵌入分别通过两个独立的线性层进行并行投影
class MLP(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        # 第一个线性层：处理emb1
        self.mlp1 = nn.Linear(d_model, d_model)
        # 第二个线性层：处理emb2
        self.mlp2 = nn.Linear(d_model, d_model)
        # （已注释）MLP-Mixer风格的多层感知机替代实现
        # self.mlp1 = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(d_model, d_model * 4)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(d_model * 4, d_model))
        # ]))
        # self.mlp2 = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(d_model, d_model * 4)),
        #     ("gelu", QuickGELU()),  
        #     ("c_proj", nn.Linear(d_model * 4, d_model))]))

    # 前向传播：返回两个嵌入各自的投影结果
    def forward(self, emb1, emb2):
        return self.mlp1(emb1), self.mlp2(emb2)

# Projector: 投影器类，维护多个独立的线性投影层，通过索引选择使用哪个投影层
class Projector(nn.Module):
    def __init__(self, dim_in, dim_out, length):
        super(Projector, self).__init__()
        # 创建length个独立的线性投影层
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(length)])
        # self.apply(weights_init)

    # 前向传播：根据索引idx选择对应的线性层进行投影
    def forward(self, emb,idx):
        return self.fc[idx](emb)

# Projector_multi_layer: 多层投影器，每个投影分支由多层全连接+激活函数组成
class Projector_multi_layer(nn.Module):
    def __init__(self, dim_in, dim_out, length, depth=1, activation=nn.ReLU):
        super(Projector_multi_layer, self).__init__()
        # 确保深度至少为1
        self.depth = max(1, depth)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation

        # 为每个索引创建独立的多层投影分支
        self.fc = nn.ModuleList([self._build_branch() for _ in range(length)])

    # _build_branch: 构建单个多层投影分支
    # 结构：Linear -> Activation -> Linear -> ... -> Linear（最后一层不加激活）
    def _build_branch(self):
        layers = []
        in_dim = self.dim_in
        # 逐层构建：每层包含一个线性变换，非最后一层后跟激活函数
        for layer_idx in range(self.depth):
            out_dim = self.dim_out
            layers.append(nn.Linear(in_dim, out_dim))
            # 如果不是最后一层，添加激活函数
            if layer_idx != self.depth - 1:
                layers.append(self.activation())
            in_dim = out_dim
        return nn.Sequential(*layers)

    # 前向传播：根据索引idx选择对应的多层投影分支
    def forward(self, emb, idx):
        return self.fc[idx](emb)

# weights_init: 权重初始化函数，根据模块类型应用不同的初始化策略
def weights_init(m):
    classname = m.__class__.__name__
    # 线性层：权重正态初始化（均值0，标准差0.02），偏置置零
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    # BatchNorm层：权重正态初始化（均值1，标准差0.02），偏置置零
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    # LayerNorm层：权重全1，偏置置零
    elif classname.find('LayerNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)
    # 卷积层：权重正态初始化（均值0，标准差0.02），偏置置零
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

# avg_pool2d_reflect: 镜像填充平均池化，保持与输入相同的空间分辨率
# 注意：此函数被定义了两次（重复），第二次定义会覆盖第一次
def avg_pool2d_reflect(x, k):
    """
    x : (B, C, H, W)
    k : kernel_size，必须为奇数才能"same"输出
    返回与 x 同分辨率的镜像平均池化
    """
    pad = k // 2  # 与零填充写法一致
    # F.pad 接收顺序是 (left, right, top, bottom)
    # 使用镜像模式进行填充
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    # 使用padding=0的平均池化（因为已在外部完成镜像填充）
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=0)

# （重复定义，实际生效的是这个版本）
def avg_pool2d_reflect(x, k):
    """
    x : (B, C, H, W)
    k : kernel_size，必须为奇数才能"same"输出
    返回与 x 同分辨率的镜像平均池化
    """
    pad = k // 2  # 与零填充写法一致
    # F.pad 接收顺序是 (left, right, top, bottom)
    # 使用镜像模式进行填充
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    # 使用padding=0的平均池化（因为已在外部完成镜像填充）
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=0)

# average_neighbor: 多尺度邻域平均池化（1x1, 3x3, 5x5）
# 对每个token做多尺度平均池化并取均值叠加，增强局部上下文信息
def average_neighbor(x, K=(1, 3, 5)):
    """
    对每个 token 做 1×1、3×3、5×5 平均池化并叠加（取均值）
    输入  : (B, N, C)  其中 N = H*W，H=W=sqrt(N)
    输出  : 同形状 (B, N, C)
    """
    B, N, C = x.shape
    H = W = int(N ** 0.5)
    # 将token序列重塑为2D特征图：[B, N, C] -> [B, C, H, W]
    x = x.transpose(1, 2).reshape(B, C, H, W)  # -> (B, C, H, W)

    outs = []
    # 对每个尺度窗口进行平均池化
    for k in K:  # 多尺度窗口
        # outs.append(avg_pool2d_reflect(x, k))
        # 使用标准平均池化（padding=k//2保持输出尺寸不变）
        outs.append(F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2))
    # 将多尺度池化结果取均值叠加
    out = sum(outs) / len(outs)  # 元素级平均叠加
    # 将特征图还原为token序列：[B, C, H, W] -> [B, N, C]
    out = out.flatten(2).transpose(1, 2)  # -> (B, N, C)
    return out
