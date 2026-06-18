# 导入进度条显示库，用于训练过程中可视化迭代进度
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
# 导入自定义 MLP 模块中的 average_neighbor 函数，用于对 patch 特征进行邻域平均
from models.mlp import average_neighbor
# 导入 numpy，用于 KMeans 聚类数据预处理和 faiss 索引构建
import numpy as np
# 导入 KMeans 聚类算法（用于记忆库构建中的聚类操作）
from sklearn.cluster import KMeans

# faiss 可选依赖：仅在 use_faiss=True 时使用
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ============================================================
# _faiss_topk_attention: Faiss Top-K 稀疏注意力辅助函数
# ============================================================
def _faiss_topk_attention(queries, keys, topk, device):
    """
    使用 Faiss IndexFlatIP 检索 Top-K 最近邻，
    仅对 Top-K 位置计算精确内积相似度，其余位置置为 -inf。
    梯度通过 PyTorch 精确相似度计算保留。

    参数:
        queries: (N_q, D) PyTorch tensor (在目标 device 上)
        keys:    (N, D) PyTorch tensor (将被移至 CPU 用于 faiss 索引)
        topk:    int, 检索的最近邻数量
        device:  torch.device, 目标计算设备

    返回:
        sparse_sim: (N_q, N) PyTorch tensor，Top-K 位置为精确相似度，其余为 -inf
    """
    # 检查 faiss 是否可用
    if not FAISS_AVAILABLE:
        raise ImportError(
            "faiss 未安装，无法使用 Top-K 稀疏注意力。"
            "请执行 pip install faiss-cpu 安装 faiss，或设置 use_faiss=False。"
        )
    # 将 keys 和 queries 转为 CPU numpy 数组（detach 断开梯度）
    keys_np = keys.detach().cpu().numpy().astype('float32')
    queries_np = queries.detach().cpu().numpy().astype('float32')

    d = keys_np.shape[1]
    # 构建 Faiss 内积搜索索引（基于 CPU 精确搜索）
    index = faiss.IndexFlatIP(d)
    index.add(keys_np)

    # 检索 Top-K 最近邻
    D_np, I_np = index.search(queries_np, topk)  # (N_q, topk)

    # 将索引转回 PyTorch tensor
    I_tensor = torch.from_numpy(I_np).long().to(device)
    # 提取 Top-K 的 keys: (N_q, topk, D)
    keys_device = keys.to(device)
    topk_keys = keys_device[I_tensor]  # (N_q, topk, D)
    # 在 PyTorch 中计算精确内积（保留梯度，queries 可反向传播）
    exact_sim = torch.einsum('qd,qkd->qk', queries, topk_keys)  # (N_q, topk)

    # 构建稀疏相似度矩阵，非 Top-K 位置填充 -inf
    N_q = queries.shape[0]
    N = keys.shape[0]
    sparse_sim = torch.full((N_q, N), float('-inf'), device=device)
    sparse_sim.scatter_(1, I_tensor, exact_sim)

    return sparse_sim


# ============================================================
# build_cache_model: 构建图像级记忆库（正常/异常分类的 keys 和 values）
# ============================================================
def build_cache_model(load_cache=False, clip_model=None, train_loader_cache=None, device=None, dir=None):
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
                labels = items['anomaly'].to(device)
                # 使用 CLIP 模型编码图像，获取图像级特征
                image_features, _, _, _ = clip_model.encode_image(images, [6, 12, 18, 24], DPAM_layer=24)
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

        # 将构建好的记忆库保存到磁盘
        torch.save({"keys": cache_keys.cpu(), "values": cache_values.cpu()}, cache_dir)

    # --- 分支2: 从磁盘加载已有记忆库 ---
    else:
        cache_dict = torch.load(cache_dir, map_location="cpu")
        cache_keys = cache_dict["keys"].to(device)
        cache_values = cache_dict["values"].to(device)
    return cache_keys, cache_values


# ============================================================
# build_patch_cache_model: 构建 patch 级记忆库
# 改进：使用 KMeans 聚类生成 K 个原型/类，替代逐样本均值（use_kmeans=True）
# ============================================================
def build_patch_cache_model(load_cache=False, clip_model=None, train_loader_cache=None,
                            device=None, dir=None,
                            use_kmeans=True, k_prototypes=8, max_kmeans_samples=50000):
    """
    构建 patch 级记忆库。

    新增参数:
        use_kmeans: bool, 是否使用 KMeans 聚类生成原型（默认 True，改进模式）
        k_prototypes: int, 每类（正常/异常）聚类的原型数（默认 8）
        max_kmeans_samples: int, KMeans 输入的最大 patch 数（防止内存溢出，默认 50000）
    """
    cache_dir = dir
    # --- 分支1: 从头构建 patch 级记忆库 ---
    if load_cache == False:    
        cache_keys = []
        cache_values = []

        # 禁用梯度计算
        with torch.no_grad():
            train_features = []
            train_labels = []

            # === 改进模式：两阶段 KMeans 聚类（先收集全部 patch，再按类别分别聚类）===
            if use_kmeans:
                # 阶段一：遍历所有训练数据，按类别收集全部 patch 特征
                pos_patches_all = []  # 异常类（类别 1）所有 patch
                neg_patches_all = []  # 正常类（类别 0）所有 patch

                for items in tqdm(train_loader_cache):
                    images = items['img'].to(device)
                    gt = items['img_mask'].squeeze().to(device)  # b 518 518
                    # 将 ground truth mask 二值化
                    gt[gt > 0.5] = 1
                    gt[gt <= 0.5] = 0
                    # 使用 CLIP 模型编码图像，获取 patch 级特征
                    image_fe, patch_features, _, patch_projections = clip_model.encode_image(images, [6, 12, 18, 24], DPAM_layer=24)
                    patch_feature = patch_features[3]
                    # 对 patch 特征进行邻域平均，增强局部上下文信息
                    patch_feature = average_neighbor(patch_feature)
                    # L2 归一化（为 spherical KMeans 做准备）
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)

                    # 将 ground truth mask 下采样到 patch 特征图尺寸 (37x37)
                    gt_resized = F.interpolate(gt.unsqueeze(1), size=(37, 37), mode='bilinear', align_corners=False)
                    gt_resized = gt_resized.squeeze(1)  # (B, 37, 37)

                    # 逐样本收集正常/异常 patch
                    for i in range(images.size(0)):
                        patch = patch_feature[i]              # (1369, D)
                        patch = patch.view(37, 37, -1)        # (37, 37, D)
                        mask = gt_resized[i]                  # (37, 37)
                        pos_mask = (mask == 1)                # abnormal
                        neg_mask = (mask == 0)                # normal

                        # 收集异常类 patch
                        if pos_mask.sum() > 0:
                            pos_patches_all.append(patch[pos_mask])  # (n_pos, D)
                        # 收集正常类 patch
                        if neg_mask.sum() > 0:
                            neg_patches_all.append(patch[neg_mask])  # (n_neg, D)

                # 阶段二：对异常类 patch 执行 KMeans 聚类生成 k 个原型
                if len(pos_patches_all) > 0:
                    pos_features = torch.cat(pos_patches_all, dim=0).cpu().numpy().astype('float32')
                    # 如果 patch 数量过多，随机采样以控制 KMeans 计算量
                    if pos_features.shape[0] > max_kmeans_samples:
                        idx = np.random.choice(pos_features.shape[0], max_kmeans_samples, replace=False)
                        pos_features = pos_features[idx]
                    # 确定实际聚类数（不能超过样本数）
                    effective_k = min(k_prototypes, pos_features.shape[0])
                    if effective_k > 1:
                        kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=42)
                        kmeans.fit(pos_features)
                        prototypes = kmeans.cluster_centers_  # (effective_k, D)
                    else:
                        # 样本不足 k 个时使用均值
                        prototypes = pos_features.mean(axis=0, keepdims=True)
                    # 转回 PyTorch 并 L2 归一化
                    prototypes = torch.from_numpy(prototypes).float().to(device)
                    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
                    train_features.append(prototypes)                                           # 异常原型
                    train_labels.append(torch.ones(effective_k, device=device))                 # 标签=1

                # 对正常类 patch 执行 KMeans 聚类生成 k 个原型
                if len(neg_patches_all) > 0:
                    neg_features = torch.cat(neg_patches_all, dim=0).cpu().numpy().astype('float32')
                    # 如果 patch 数量过多，随机采样
                    if neg_features.shape[0] > max_kmeans_samples:
                        idx = np.random.choice(neg_features.shape[0], max_kmeans_samples, replace=False)
                        neg_features = neg_features[idx]
                    # 确定实际聚类数
                    effective_k = min(k_prototypes, neg_features.shape[0])
                    if effective_k > 1:
                        kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=42)
                        kmeans.fit(neg_features)
                        prototypes = kmeans.cluster_centers_
                    else:
                        prototypes = neg_features.mean(axis=0, keepdims=True)
                    # 转回 PyTorch 并 L2 归一化
                    prototypes = torch.from_numpy(prototypes).float().to(device)
                    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
                    train_features.append(prototypes)                                           # 正常原型
                    train_labels.append(torch.zeros(effective_k, device=device))                 # 标签=0

            # === 原始模式：逐样本均值（向后兼容，use_kmeans=False 时使用）===
            else:
                for items in tqdm(train_loader_cache):
                    images = items['img'].to(device)
                    labels = items['anomaly'].to(device)
                    gt = items['img_mask'].squeeze().to(device)
                    gt[gt > 0.5] = 1
                    gt[gt <= 0.5] = 0
                    image_fe, patch_features, _, patch_projections = clip_model.encode_image(images, [6, 12, 18, 24], DPAM_layer=24)
                    patch_feature = patch_features[3]
                    patch_feature = average_neighbor(patch_feature)

                    gt_resized = F.interpolate(gt.unsqueeze(1), size=(37, 37), mode='bilinear', align_corners=False)
                    gt_resized = gt_resized.squeeze(1)

                    # 逐样本处理：根据 mask 将 patch 分为异常和正常两类
                    for i in range(images.size(0)):
                        patch = patch_feature[i]              # (1369, 768)
                        patch = patch.view(37, 37, -1)        # (37, 37, 768)
                        mask = gt_resized[i]                 # (37, 37)
                        pos_mask = (mask == 1)               # abnormal
                        neg_mask = (mask == 0)               # normal

                        # 如果存在异常 patch，计算其均值特征并加入记忆库（标签=1）
                        if pos_mask.sum() > 0:
                            pos_feat = patch[pos_mask]
                            pos_feat = pos_feat.mean(dim=0, keepdim=True)
                            pos_feat = pos_feat / pos_feat.norm(dim=-1, keepdim=True)
                            train_features.append(pos_feat)
                            train_labels.append(torch.tensor([1], device=device))

                        # 如果存在正常 patch，计算其均值特征并加入记忆库（标签=0）
                        if neg_mask.sum() > 0:
                            neg_feat = patch[neg_mask]
                            neg_feat = neg_feat.mean(dim=0, keepdim=True)
                            neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)
                            train_features.append(neg_feat)
                            train_labels.append(torch.tensor([0], device=device))

            # 将所有原型/样本特征拼接
            cache_keys = torch.cat(train_features, dim=0)
            raw_labels = torch.cat(train_labels, dim=0).to(torch.int64)
            # 将标签转为 one-hot 编码
            cache_values = F.one_hot(raw_labels, num_classes=2).float().to(device)
        cache_dict = {
            "keys": cache_keys,
            "values": cache_values
        }

        # 保存 patch 级记忆库到磁盘
        torch.save({"keys": cache_keys.cpu(), "values": cache_values.cpu()}, cache_dir)

    # --- 分支2: 从磁盘加载已有 patch 级记忆库 ---
    else:
        cache_dict = torch.load(cache_dir, map_location="cpu")
        cache_keys = cache_dict["keys"].to(device)
        cache_values = cache_dict["values"].to(device)
    return cache_keys, cache_values


# ============================================================
# compute_socre: 图像级分类评分（基于记忆库的相似度检索）
# 改进：use_faiss=True 时使用 Faiss Top-K 稀疏注意力替代全量 softmax(QK^T)
# ============================================================
def compute_socre(image_features, cache_keys, cache_values, device, proj=None,
                  need_mask=False, is_train=False, use_proj=True,
                  use_faiss=False, faiss_topk=20):
    """
    图像级分类评分。

    新增参数:
        use_faiss: bool, 是否使用 Faiss Top-K 稀疏注意力（默认 False，保持兼容）
        faiss_topk: int, Faiss 检索的最近邻数量（默认 20）
    """
    # 计算图像特征与记忆库 keys 的原始余弦相似度（始终全量计算，用于 mask 阈值）
    ori_sim_weights = torch.matmul(image_features, cache_keys.to(device).t())  # b n
    loss_keys = torch.tensor(0.0, device=device)

    # 如果启用投影适配器，计算投影后的相似度
    if use_proj and proj is not None:
        image_features_proj, cache_keys_proj = proj(image_features, cache_keys)

        # === Faiss Top-K 稀疏注意力（可选）===
        if use_faiss:
            # 使用 Faiss 检索 Top-K 最近邻，构建稀疏相似度矩阵
            sim_weights = _faiss_topk_attention(image_features_proj, cache_keys_proj, faiss_topk, device)
        else:
            # 原始全量 matmul
            sim_weights = torch.matmul(image_features_proj, cache_keys_proj.to(device).t())  # b n
    else:
        sim_weights = ori_sim_weights

    # 如果启用 mask 机制，过滤掉相似度过高的样本（防止过拟合）
    if need_mask:
        # 取原始相似度 95% 分位数作为阈值
        th = torch.quantile(ori_sim_weights, 0.95, dim=-1, keepdim=True)
        mask = ori_sim_weights > th
        mask_counts = mask.sum(dim=1)
        # 将高于阈值的相似度置为 -inf，softmax 后权重趋近于 0
        sim_weights = sim_weights.masked_fill(mask, float('-inf'))
    # 对相似度权重做 softmax 归一化
    sim_weights = F.softmax(sim_weights, dim=-1)
    # 用归一化后的权重加权求和记忆库 values，得到分类 logits
    logits = torch.matmul(sim_weights, cache_values.to(device).float())
    return logits, loss_keys


# ============================================================
# compute_patch_socre: patch 级分割评分（基于记忆库的 patch 相似度检索）
# 改进：use_faiss=True 时使用 Faiss Top-K 稀疏注意力替代全量 softmax(QK^T)
# ============================================================
def compute_patch_socre(patch_features, cache_keys, cache_values, ori_sim_weights=None,
        device=None, proj=None, need_mask=False, patch_projection=False, gt_mask=None,
        anomaly_threshold=0.5, is_mradft=False, use_proj=True,
        use_faiss=False, faiss_topk=50):
    """
    patch 级分割评分。

    新增参数:
        use_faiss: bool, 是否使用 Faiss Top-K 稀疏注意力（默认 False，保持兼容）
        faiss_topk: int, Faiss 检索的最近邻数量（默认 50）
    """
    # 计算每个 patch 特征与记忆库 keys 的原始相似度（始终全量计算，用于 mask 阈值和 finetune_sim_weights）
    ori_sim_weights = torch.matmul(patch_features, cache_keys.to(device).t())  # b 1369 n

    # 如果启用投影适配器，分别对 patch 特征和记忆库 keys 投影后再计算相似度
    if use_proj and proj is not None:
        patch_features_proj = proj(patch_features, 0)
        cache_keys_proj = proj(cache_keys, 1)

        # === Faiss Top-K 稀疏注意力（可选）===
        if use_faiss:
            # 将 patch 特征的 batch 维度展平为 (B*1369, D)
            B, N_p, D = patch_features_proj.shape
            flat_queries = patch_features_proj.reshape(B * N_p, D)
            # 使用 Faiss 检索 Top-K 最近邻
            flat_sim = _faiss_topk_attention(flat_queries, cache_keys_proj, faiss_topk, device)
            # 恢复 batch 维度: (B*1369, N) → (B, 1369, N)
            sim_weights = flat_sim.reshape(B, N_p, -1)
        else:
            # 原始全量 matmul
            sim_weights = torch.matmul(patch_features_proj, cache_keys_proj.T.to(device))  # b 1369 n
    else:
        sim_weights = ori_sim_weights

    # 保存微调阶段的相似度副本
    finetune_sim_weights = sim_weights.clone()
    # 如果启用 mask 机制，过滤掉相似度过高的 patch（防止过拟合）
    if need_mask:
        # 取原始相似度 80% 分位数作为阈值
        th = torch.quantile(ori_sim_weights, 0.8, dim=-1, keepdim=True)
        mask = ori_sim_weights > th
        mask_counts = mask.sum(dim=1)
        # 将高于阈值的相似度置为 -inf
        sim_weights = sim_weights.masked_fill(mask, float('-inf'))

    # 对相似度权重做 softmax 归一化
    sim_weights = F.softmax(sim_weights, dim=-1)
    # 加权求和得到每个 patch 的正常/异常 logits
    logits = torch.matmul(sim_weights, cache_values.to(device).float())  # (b, 1369, 2)

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
    return logits, new_patch_features, ori_sim_weights, finetune_sim_weights
