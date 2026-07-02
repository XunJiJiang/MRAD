# 导入进度条显示库，用于训练过程中可视化迭代进度
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
# 导入自定义 MLP 模块中的 average_neighbor 函数，用于对 patch 特征进行邻域平均
from models.mlp import average_neighbor
import torch
import torch.nn.functional as F
from tqdm import tqdm
# 导入 numpy 用于压缩算法中的数值计算（CPU 侧操作，避免 GPU OOM）
import numpy as np
# 导入 MiniBatchKMeans 聚类算法（用于记忆库压缩中的 KMeans 方法）
from sklearn.cluster import MiniBatchKMeans


# ============================================================
# 记忆库压缩模块：支持 KMeans / Greedy Prototype / Herding 三种方法
# 论文动机：原 Memory Bank 存在大量冗余（作者 ablation 证明 Memory 减少很多性能几乎不掉），
#           因此先将全部样本压缩为代表性原型，再进行检索，几乎不改主网络。
# 流程： Original Memory → Representative Prototype → Retrieval
# ============================================================

def _kmeans_compress(keys_np, n_prototypes, seed=42):
    """KMeans 聚类压缩：对每类样本做 MiniBatchKMeans，取聚类中心作为代表性原型。

    Args:
        keys_np: (N, D) numpy float32 数组，单类样本特征
        n_prototypes: 目标原型数量
        seed: 随机种子，保证可复现

    Returns:
        (n_prototypes, D) numpy 数组，聚类中心
    """
    # 确保聚类数不超过实际样本数
    n_actual = min(n_prototypes, len(keys_np))
    # 使用 MiniBatchKMeans，batch_size 自适应，适合大规模记忆库
    kmeans = MiniBatchKMeans(
        n_clusters=n_actual,
        batch_size=min(2048, len(keys_np)),
        random_state=seed,
        max_iter=100,
        n_init=3
    )
    # 在 CPU 上拟合，避免 GPU 显存溢出
    kmeans.fit(keys_np)
    # 聚类中心是簇内均值，尚未归一化
    return kmeans.cluster_centers_


def _greedy_compress(keys_np, n_prototypes, seed=42):
    """Greedy k-Center 压缩（最远点遍历）：贪心选择覆盖最大范围的代表性原型。

    从最接近均值的点开始，每次选择距离已选集合最远的点，直到选满 n_prototypes 个。

    Args:
        keys_np: (N, D) numpy float32 数组，单类样本特征
        n_prototypes: 目标原型数量
        seed: 随机种子（此方法为确定性算法，seed 保留用于接口一致性）

    Returns:
        (n_prototypes, D) numpy 数组，选中的原型
    """
    n_actual = min(n_prototypes, len(keys_np))

    # 从最接近全局均值的点开始（确保起始点具有代表性）
    mean = keys_np.mean(axis=0)
    dists_to_mean = np.linalg.norm(keys_np - mean, axis=1)
    first_idx = int(np.argmin(dists_to_mean))

    selected = [first_idx]
    # min_dist[i] = 样本 i 到已选集合的最近距离
    min_dist = np.linalg.norm(keys_np - keys_np[first_idx], axis=1)

    # 贪心选取：每次取 argmax(min_dist) 作为下一个原型
    for _ in range(1, n_actual):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        # 更新各样本到已选集合的最近距离
        new_dist = np.linalg.norm(keys_np - keys_np[next_idx], axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    return keys_np[selected]


def _herding_compress(keys_np, n_prototypes, seed=42):
    """Herding 压缩：迭代选择使运行均值逼近全局均值的样本。

    每步选择 <x, t·μ - w_t> 最大的点，其中 μ 为全局均值，w_t 为已选点之和，
    使得前 t 个原型的均值尽可能接近全部样本的均值。

    Args:
        keys_np: (N, D) numpy float32 数组，单类样本特征
        n_prototypes: 目标原型数量
        seed: 随机种子（此方法为确定性算法，seed 保留用于接口一致性）

    Returns:
        (n_prototypes, D) numpy 数组，选中的原型
    """
    n_actual = min(n_prototypes, len(keys_np))

    # 全局均值 μ
    mu = keys_np.mean(axis=0)
    selected = []
    # w_t = 已选原型的累加和
    w_t = np.zeros_like(mu)

    # 逐步选取：第 t 步选择使 <x, t·μ - w_t> 最大的样本
    for t in range(1, n_actual + 1):
        score = keys_np @ (t * mu - w_t)
        # 排除已选样本
        score[selected] = -np.inf
        idx = int(np.argmax(score))
        selected.append(idx)
        w_t += keys_np[idx]

    return keys_np[selected]


def compress_memory(cache_keys, cache_values, method='none', n_prototypes=500, device=None):
    """对记忆库进行类别感知的压缩：正常类和异常类分别压缩。

    论文 ablation 表明 Memory 大幅减少后性能几乎不掉，说明存在大量冗余。
    本函数将原始记忆库压缩为代表性原型，减少检索开销同时保留判别信息。

    Args:
        cache_keys: (N, D) tensor — L2 归一化的记忆库 keys
        cache_values: (N, 2) tensor — one-hot 标签 [normal, anomaly]
        method: 压缩方法 — 'none'（不压缩）、'kmeans'、'greedy'、'herding'
        n_prototypes: 每类压缩后的原型数量（如 500 表示正常 500 + 异常 500 = 1000 总计）
        device: 输出 tensor 的目标设备

    Returns:
        compressed_keys: (M, D) tensor — 压缩后的 keys
        compressed_values: (M, 2) tensor — 对应的 one-hot 标签
    """
    # 不压缩时直接返回原始记忆库
    if method == 'none' or method is None:
        return cache_keys, cache_values

    if device is None:
        device = cache_keys.device

    # 从 one-hot 标签中提取类别索引：0=正常，1=异常
    labels = cache_values.argmax(dim=1)

    compressed_keys_list = []
    compressed_labels_list = []

    # 逐类压缩，保持标签信息
    for cls in [0, 1]:
        # 提取当前类的所有样本
        mask = (labels == cls)
        cls_keys = cache_keys[mask]
        n_cls = len(cls_keys)

        # 该类无样本则跳过
        if n_cls == 0:
            continue

        # 转换为 CPU numpy 数组进行压缩计算（避免 GPU OOM）
        cls_keys_np = cls_keys.cpu().numpy().astype(np.float32)

        # 样本数不足时直接保留全部，不压缩
        if n_cls <= n_prototypes:
            protos_np = cls_keys_np
        else:
            # 根据指定方法进行压缩
            if method == 'kmeans':
                protos_np = _kmeans_compress(cls_keys_np, n_prototypes)
            elif method == 'greedy':
                protos_np = _greedy_compress(cls_keys_np, n_prototypes)
            elif method == 'herding':
                protos_np = _herding_compress(cls_keys_np, n_prototypes)
            else:
                raise ValueError(f"未知的压缩方法: {method}，可选: none/kmeans/greedy/herding")

        # 转回 tensor 并 L2 归一化（KMeans 中心是均值，需要重新归一化；
        # greedy/herding 选取的点本身已归一化，但统一归一化更安全）
        protos = torch.from_numpy(protos_np).to(dtype=cache_keys.dtype, device=device)
        protos = protos / protos.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        compressed_keys_list.append(protos)
        compressed_labels_list.append(
            torch.full((len(protos),), cls, device=device, dtype=torch.int64)
        )

    # 所有类均无样本的极端情况，返回原始记忆库
    if len(compressed_keys_list) == 0:
        return cache_keys, cache_values

    # 拼接所有类的压缩结果
    new_keys = torch.cat(compressed_keys_list, dim=0)
    new_labels = torch.cat(compressed_labels_list, dim=0)
    new_values = F.one_hot(new_labels, num_classes=2).float()

    print(f"[Memory Compression] method={method}, n_prototypes_per_class={n_prototypes}, "
          f"original={len(cache_keys)} → compressed={len(new_keys)}")

    return new_keys, new_values


# ============================================================
# build_cache_model: 构建图像级记忆库（正常/异常分类的 keys 和 values）
# ============================================================
def build_cache_model(load_cache = False,  clip_model = None, train_loader_cache = None,
                      device = None, dir=None,
                      compress_method='none', n_prototypes=500):
    cache_dir = dir
    # --- 分支1: 从头构建记忆库 ---
    if load_cache == False:    
        cache_keys = []
        cache_values = []

        # 禁用梯度计算，加速推理并节省显存
        with torch.no_grad():
            # Data augmentation for the cache model
            train_features = []
            train_labels = []
            # 遍历训练数据加载器，逐批提取图像特征
            for items in tqdm(train_loader_cache):
                images = items['img'].to(device)
                labels =  items['anomaly'].to(device)
                # 使用 CLIP 模型编码图像，获取图像级特征
                image_features,_ ,_,_= clip_model.encode_image(images,[6, 12, 18, 24],DPAM_layer=24)
                # 对特征进行 L2 归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                train_features.append(image_features)
                train_labels.append(labels)
            # 将所有 batch 的特征和标签拼接成完整矩阵
            cache_keys = torch.cat(train_features, dim=0)
            raw_labels = torch.cat(train_labels, dim=0).to(torch.int64)
            # 将标签转换为 one-hot 编码（2 类：正常/异常）
            cache_values = F.one_hot(raw_labels, num_classes=2).float().to(device)
        cache_dict = {
            "keys": cache_keys,
            "values": cache_values
        }

        # 将构建好的完整记忆库保存到磁盘（始终保存未压缩版本，便于不同压缩实验复用）
        torch.save({"keys": cache_keys.cpu(), "values": cache_values.cpu()}, cache_dir)

    # --- 分支2: 从磁盘加载已有记忆库 ---
    else:
        cache_dict = torch.load(cache_dir, map_location="cpu")
        cache_keys = cache_dict["keys"].to(device)
        cache_values = cache_dict["values"].to(device)

    # --- 记忆库压缩（在构建或加载之后统一应用，on-the-fly 压缩） ---
    # 磁盘 .pt 文件始终存储完整未压缩记忆库；此处根据参数动态压缩
    # 支持不同实验复用同一 .pt 文件，只需修改压缩参数即可
    if compress_method != 'none':
        cache_keys, cache_values = compress_memory(
            cache_keys, cache_values,
            method=compress_method,
            n_prototypes=n_prototypes,
            device=device
        )

    return cache_keys, cache_values


# ============================================================
# build_patch_cache_model: 构建 patch 级记忆库（逐样本收集正常/异常 patch 的均值特征）
# ============================================================
def build_patch_cache_model(load_cache = False,  clip_model = None, train_loader_cache = None,
                            device = None, dir=None,
                            compress_method='none', n_prototypes=500):
    cache_dir = dir
    # --- 分支1: 从头构建 patch 级记忆库 ---
    if load_cache == False:    
        cache_keys = []
        cache_values = []

        # 禁用梯度计算
        with torch.no_grad():
            # Data augmentation for the cache model
            train_features = []
            train_labels = []
            # 遍历训练数据，逐样本提取 patch 特征
            for items in tqdm(train_loader_cache):
                images = items['img'].to(device)
                labels =  items['anomaly'].to(device)# b
                gt = items['img_mask'].squeeze().to(device) # b 518 518
                # 将 ground truth mask 二值化：>0.5 视为异常，<=0.5 视为正常
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                # 使用 CLIP 模型编码图像，同时获取图像级特征和 patch 级特征
                image_fe,patch_features ,_,patch_projections = clip_model.encode_image(images,[6, 12, 18, 24],DPAM_layer=24)
                patch_feature = patch_features[3]
                # 对 patch 特征进行邻域平均，增强局部上下文信息
                patch_feature = average_neighbor(patch_feature)
                # patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True) #b 1369 1024
                #
                # 将 ground truth mask 下采样到 patch 特征图尺寸 (37x37)
                gt_resized = F.interpolate(gt.unsqueeze(1), size=(37, 37), mode='bilinear', align_corners=False)
                gt_resized = gt_resized.squeeze(1)  # (B, 37, 37)

                # 逐样本处理：根据 mask 将 patch 分为异常和正常两类
                for i in range(images.size(0)):
                    patch = patch_feature[i]              # (1369, 768)
                    patch = patch.view(37, 37, -1)        # (37, 37, 768)
                    mask = gt_resized[i]                 # (37, 37)

                    pos_mask = (mask == 1)               # abnormal
                    neg_mask = (mask == 0)               # normal

                    # 如果存在异常 patch，计算其均值特征并加入记忆库（标签=1）
                    if pos_mask.sum() > 0:
                        pos_feat = patch[pos_mask]    # (n_pos, 768)
                        pos_feat = pos_feat.mean(dim = 0, keepdim=True)
                        pos_feat = pos_feat / pos_feat.norm(dim=-1, keepdim=True)
                        train_features.append(pos_feat)  # anomaly → 1
                        train_labels.append(torch.tensor([1], device=device))  # anomaly → 1

                    # 如果存在正常 patch，计算其均值特征并加入记忆库（标签=0）
                    if neg_mask.sum() > 0:
                        neg_feat = patch[neg_mask]      # (n_neg, 768)
                        neg_feat = neg_feat.mean(dim = 0, keepdim=True)
                        neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)
                        train_features.append(neg_feat)  # normal → 0
                        train_labels.append(torch.tensor([0], device=device))  # normal → 0

            # 将所有样本的 patch 特征拼接
            cache_keys = torch.cat(train_features, dim=0)
            raw_labels = torch.cat(train_labels, dim=0).to(torch.int64)
            # 将标签转为 one-hot 编码
            cache_values = F.one_hot(raw_labels, num_classes=2).float().to(device)
        cache_dict = {
            "keys": cache_keys,
            "values": cache_values
        }

        # 保存完整的 patch 级记忆库到磁盘（始终保存未压缩版本，便于不同压缩实验复用）
        torch.save({"keys": cache_keys.cpu(), "values": cache_values.cpu()}, cache_dir)

    # --- 分支2: 从磁盘加载已有 patch 级记忆库 ---
    else:
        cache_dict = torch.load(cache_dir, map_location="cpu")
        cache_keys = cache_dict["keys"].to(device)
        cache_values = cache_dict["values"].to(device)

    # --- 记忆库压缩（在构建或加载之后统一应用，on-the-fly 压缩） ---
    # 与 build_cache_model 一致：磁盘 .pt 文件存储完整未压缩记忆库，
    # 此处根据参数动态压缩，支持不同实验复用同一 .pt 文件
    if compress_method != 'none':
        cache_keys, cache_values = compress_memory(
            cache_keys, cache_values,
            method=compress_method,
            n_prototypes=n_prototypes,
            device=device
        )

    return cache_keys, cache_values

# ============================================================
# compute_socre: 图像级分类评分（基于记忆库的相似度检索）
# ============================================================
def compute_socre(image_features, cache_keys, cache_values, device, proj=None, need_mask=False, is_train=False, use_proj=True):
    # scale = 768**-0.5
    # 计算图像特征与记忆库 keys 的原始余弦相似度
    ori_sim_weights = torch.matmul(image_features, cache_keys.to(device).t())#b n
    loss_keys = torch.tensor(0.0, device=device)

    # 如果启用投影适配器，对特征进行投影后再计算相似度
    if use_proj and proj is not None:
        image_features_proj, cache_keys_proj = proj(image_features, cache_keys)
        sim_weights = torch.matmul(image_features_proj, cache_keys_proj.to(device).t())#b n
    else:
        sim_weights = ori_sim_weights

    # 如果启用 mask 机制，过滤掉相似度过高的样本（防止过拟合）
    if need_mask:
        # 取原始相似度 95% 分位数作为阈值
        th = torch.quantile(ori_sim_weights, 0.95, dim=-1, keepdim=True)
        mask = ori_sim_weights>th    #default 0.9
        mask_counts = mask.sum(dim=1)
        # print(mask_counts) 
        # print(mask.nonzero())
        # 将高于阈值的相似度置为 -inf，softmax 后权重趋近于 0
        sim_weights = sim_weights.masked_fill(mask, float('-inf'))
    # 对相似度权重做 softmax 归一化
    sim_weights = F.softmax(sim_weights, dim=-1)
    # sim_weights = 0.005*torch.exp((sim_weights-1))
    # 用归一化后的权重加权求和记忆库 values，得到分类 logits
    logits = torch.matmul(sim_weights, cache_values.to(device).float())
    return logits, loss_keys

# ============================================================
# compute_patch_socre: patch 级分割评分（基于记忆库的 patch 相似度检索）
# ============================================================
def compute_patch_socre(patch_features, cache_keys, cache_values, ori_sim_weights=None,
        device=None, proj=None, need_mask=False, patch_projection=False, gt_mask=None,
        anomaly_threshold=0.5, is_mradft=False, use_proj=True):

    # 计算每个 patch 特征与记忆库 keys 的原始相似度
    ori_sim_weights = torch.matmul(patch_features, cache_keys.to(device).t())#b 1369 n

    # 如果启用投影适配器，分别对 patch 特征和记忆库 keys 投影后再计算相似度
    if use_proj and proj is not None:
        patch_features_proj = proj(patch_features, 0)
        cache_keys_proj = proj(cache_keys, 1)
        sim_weights = torch.matmul(patch_features_proj, cache_keys_proj.T.to(device))# b 1369 n
    else:
        sim_weights = ori_sim_weights

    # 保存微调阶段的相似度副本
    finetune_sim_weights = sim_weights.clone()
    # 如果启用 mask 机制，过滤掉相似度过高的 patch（防止过拟合）
    if need_mask:
        # 取原始相似度 80% 分位数作为阈值
        th = torch.quantile(ori_sim_weights, 0.8, dim=-1, keepdim=True)
        mask = ori_sim_weights > th#test mvtec     when test visa be setted  0.85 memclip be setted 0.95
        mask_counts = mask.sum(dim=1)
        # print(mask_counts)
        # mask = mask.unsqueeze(1).expand(-1, patch_features.size(1), -1)
        # 将高于阈值的相似度置为 -inf
        sim_weights = sim_weights.masked_fill(mask, float('-inf')) 
    # similary_sum = torch.matmul(sim_weights, cache_values.to(device).float())# b 1369 2

    # 对相似度权重做 softmax 归一化
    sim_weights = F.softmax(sim_weights, dim=-1)
    # 加权求和得到每个 patch 的正常/异常 logits
    logits = torch.matmul(sim_weights, cache_values.to(device).float())  # (b, 1369, 2)

    # anomaly_weights = logits[:, :, 1]  # (b, 1369)
    # anomaly_weights = logits.permute(0, 2, 1)  # (b,2, 1369)
    # new_weights = anomaly_weights*(anomaly_weights.softmax(dim=-1)) # (b, 2, 1369)test visa
    # new_weights = anomaly_weights*(anomaly_weights/anomaly_weights.sum(dim=-1, keepdim=True))# (b, 2, 1369)test mvtec
    # new_patch_features = torch.matmul(new_weights, patch_projection)  # (b, 2, 768)
    # new_patch_features = 0


    # 如果不是 MRAD 微调模式，计算异常和正常的 token 特征用于后续处理
    if not is_mradft:
        # 提取异常概率（logits 中第 1 类为异常）
        anomaly_probs = logits[:, :, 1]
        anomaly_threshold = torch.tensor(anomaly_threshold, device=anomaly_probs.device, dtype=anomaly_probs.dtype)
        # 根据阈值划分异常区域和正常区域
        anomaly_area = (anomaly_probs > anomaly_threshold).float()  # (B, 1369)
        normal_area  = 1.0 - anomaly_area

     
        anomaly_mask = anomaly_area.unsqueeze(-1)  # (B, 1369, 1)
        normal_mask  = normal_area.unsqueeze(-1)

        # 分别计算异常区域和正常区域的 patch 投影特征之和
        anomaly_feat_sum = (patch_projection * anomaly_mask).sum(dim=1)  # (B, 768)
        normal_feat_sum  = (patch_projection * normal_mask).sum(dim=1)

        # 统计异常/正常 patch 数量，并 clamp 防止除零
        anomaly_count = anomaly_mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        normal_count  = normal_mask.sum(dim=1).clamp(min=1.0)

        # 计算异常和正常的平均 token 特征
        anomaly_token = anomaly_feat_sum / anomaly_count  # (B, 768)
        normal_token  = normal_feat_sum / normal_count

        # 如果没有异常 patch，则将异常 token 置零
        anomaly_token[anomaly_mask.sum(dim=1).squeeze(-1) == 0] = 0.0
        # 如果没有正常 patch，则将正常 token 置零
        normal_token[normal_mask.sum(dim=1).squeeze(-1) == 0] = 0.0

        # 将正常和异常 token 堆叠为 (B, 2, 768) 的新 patch 特征
        new_patch_features = torch.stack([normal_token, anomaly_token], dim=1)  # (B, 2, 768)
    # 如果是 MRAD 微调模式，不计算 new_patch_features
    else:
        new_patch_features = 0
        # print("new_patch_features is 0")
    return logits,new_patch_features,ori_sim_weights,finetune_sim_weights
