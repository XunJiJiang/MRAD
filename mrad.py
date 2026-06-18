# 导入进度条显示库，用于训练过程中可视化迭代进度
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
# 导入自定义 MLP 模块中的 average_neighbor 函数，用于对 patch 特征进行邻域平均
from models.mlp import average_neighbor
import numpy as np
# 导入 KMeans 聚类算法（用于记忆库构建中的聚类操作）
from sklearn.cluster import KMeans

# Faiss 可选导入（用于 Top-K 稀疏注意力加速）
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ============================================================
# _faiss_topk_attention: Faiss Top-K 稀疏注意力辅助函数
#   使用 CPU faiss IndexFlatIP 检索 Top-K，PyTorch einsum 回算
#   精确内积以保留梯度（仅 Top-K 位置有值，其余为 -inf）
# ============================================================
def _faiss_topk_attention(queries, keys, topk, device):
    # 检查 faiss 是否已安装
    if not FAISS_AVAILABLE:
        raise ImportError(
            "faiss 未安装，无法使用 Top-K 稀疏注意力。"
            "请执行 pip install faiss-cpu 安装 faiss，或设置 use_faiss=False。"
        )
    # 将 keys 和 queries 转为 CPU numpy，供 faiss 检索使用
    keys_np = keys.detach().cpu().numpy().astype('float32')
    queries_np = queries.detach().cpu().numpy().astype('float32')
    d = keys_np.shape[1]
    # 构建 faiss 内积索引（等价于余弦相似度，前提是特征已 L2 归一化）
    index = faiss.IndexFlatIP(d)
    index.add(keys_np)
    # 检索每个 query 的 Top-K 最近邻，返回距离和内积
    D_np, I_np = index.search(queries_np, topk)
    # 将检索到的索引转回 PyTorch tensor
    I_tensor = torch.from_numpy(I_np).long().to(device)
    # 用 PyTorch einsum 对 Top-K 位置回算精确内积（保留梯度）
    keys_device = keys.to(device)
    topk_keys = keys_device[I_tensor]           # (N_q, topk, d)
    exact_sim = torch.einsum('qd,qkd->qk', queries, topk_keys)  # (N_q, topk)
    # 构建稀疏相似度矩阵：非 Top-K 位置填充 -inf
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
#   支持两种模式：
#     use_kmeans=False (默认): 每个样本的异常/正常 patch 取均值作为原型
#     use_kmeans=True: 收集所有样本的 patch 后用 KMeans 聚类，生成 K 个原型
# ============================================================
def build_patch_cache_model(load_cache=False, clip_model=None, train_loader_cache=None, device=None, dir=None,
                            use_kmeans=False, k_prototypes=8, max_kmeans_samples=50000):
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
            # KMeans 模式下：预先收集所有 patch 的 CPU numpy 数组（避免 GPU OOM）
            pos_patches_all = []   # 异常类 patch 列表
            neg_patches_all = []   # 正常类 patch 列表
            # 遍历训练数据，逐样本提取 patch 特征
            for items in tqdm(train_loader_cache):
                images = items['img'].to(device)
                labels = items['anomaly'].to(device)       # b
                gt = items['img_mask'].squeeze().to(device) # b 518 518
                # 将 ground truth mask 二值化：>0.5 视为异常，<=0.5 视为正常
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                # 使用 CLIP 模型编码图像，同时获取图像级特征和 patch 级特征
                image_fe, patch_features, _, patch_projections = clip_model.encode_image(images, [6, 12, 18, 24], DPAM_layer=24)
                patch_feature = patch_features[3]
                # 对 patch 特征进行邻域平均，增强局部上下文信息
                patch_feature = average_neighbor(patch_feature)
                # patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True) #b 1369 1024

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

                    # --- KMeans 模式：收集每个异常 patch 到 CPU numpy 列表 ---
                    if use_kmeans:
                        if pos_mask.sum() > 0:
                            # 立即转为 CPU numpy，避免 GPU torch.cat 时 OOM
                            pos_patches_all.append(patch[pos_mask].cpu().numpy().astype('float32'))
                        if neg_mask.sum() > 0:
                            neg_patches_all.append(patch[neg_mask].cpu().numpy().astype('float32'))
                    # --- 原始模式：每个样本取均值作为单个原型 ---
                    else:
                        if pos_mask.sum() > 0:
                            pos_feat = patch[pos_mask]    # (n_pos, 768)
                            pos_feat = pos_feat.mean(dim=0, keepdim=True)
                            pos_feat = pos_feat / pos_feat.norm(dim=-1, keepdim=True)
                            train_features.append(pos_feat)  # anomaly → 1
                            train_labels.append(torch.tensor([1], device=device))  # anomaly → 1

                        if neg_mask.sum() > 0:
                            neg_feat = patch[neg_mask]      # (n_neg, 768)
                            neg_feat = neg_feat.mean(dim=0, keepdim=True)
                            neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)
                            train_features.append(neg_feat)  # normal → 0
                            train_labels.append(torch.tensor([0], device=device))  # normal → 0

            # ============================================================
            # KMeans 聚类后处理：对每类独立聚类，用聚类中心作为原型
            # ============================================================
            if use_kmeans:
                # 处理异常类 (pos) patches
                if len(pos_patches_all) > 0:
                    # 使用 np.concatenate 在 CPU 上拼接（避免 GPU OOM）
                    pos_features = np.concatenate(pos_patches_all, axis=0)
                    # 子采样防止 KMeans 内存/时间溢出
                    if pos_features.shape[0] > max_kmeans_samples:
                        idx = np.random.choice(pos_features.shape[0], max_kmeans_samples, replace=False)
                        pos_features = pos_features[idx]
                    # L2 归一化实现 spherical KMeans（基于余弦相似度聚类）
                    pos_norms = np.linalg.norm(pos_features, axis=-1, keepdims=True)
                    pos_norms = np.maximum(pos_norms, 1e-8)
                    pos_features = pos_features / pos_norms
                    # 初始化有效聚类数（不超过样本数）
                    effective_k = min(k_prototypes, pos_features.shape[0])
                    if effective_k > 1:
                        kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=42)
                        kmeans.fit(pos_features)
                        prototypes = kmeans.cluster_centers_
                    else:
                        prototypes = pos_features.mean(axis=0, keepdims=True)
                    # 对聚类中心再做 L2 归一化，保持单位球面上的一致性
                    proto_norms = np.linalg.norm(prototypes, axis=-1, keepdims=True)
                    proto_norms = np.maximum(proto_norms, 1e-8)
                    prototypes = prototypes / proto_norms
                    prototypes = torch.from_numpy(prototypes).float().to(device)
                    train_features.append(prototypes)
                    train_labels.append(torch.ones(effective_k, device=device))

                # 处理正常类 (neg) patches
                if len(neg_patches_all) > 0:
                    neg_features = np.concatenate(neg_patches_all, axis=0)
                    if neg_features.shape[0] > max_kmeans_samples:
                        idx = np.random.choice(neg_features.shape[0], max_kmeans_samples, replace=False)
                        neg_features = neg_features[idx]
                    neg_norms = np.linalg.norm(neg_features, axis=-1, keepdims=True)
                    neg_norms = np.maximum(neg_norms, 1e-8)
                    neg_features = neg_features / neg_norms
                    effective_k = min(k_prototypes, neg_features.shape[0])
                    if effective_k > 1:
                        kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=42)
                        kmeans.fit(neg_features)
                        prototypes = kmeans.cluster_centers_
                    else:
                        prototypes = neg_features.mean(axis=0, keepdims=True)
                    proto_norms = np.linalg.norm(prototypes, axis=-1, keepdims=True)
                    proto_norms = np.maximum(proto_norms, 1e-8)
                    prototypes = prototypes / proto_norms
                    prototypes = torch.from_numpy(prototypes).float().to(device)
                    train_features.append(prototypes)
                    train_labels.append(torch.zeros(effective_k, device=device))

            # 将所有样本的 patch 特征拼接
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
#   新增参数：
#     use_faiss: 是否启用 Faiss Top-K 稀疏注意力
#     faiss_topk: Top-K 的 K 值（仅当 use_faiss=True 时生效）
# ============================================================
def compute_socre(image_features, cache_keys, cache_values, device, proj=None,
                  need_mask=False, is_train=False, use_proj=True,
                  use_faiss=False, faiss_topk=20):
    # scale = 768**-0.5
    # 计算图像特征与记忆库 keys 的原始余弦相似度
    ori_sim_weights = torch.matmul(image_features, cache_keys.to(device).t())  # b n
    loss_keys = torch.tensor(0.0, device=device)

    # 如果启用投影适配器，对特征进行投影后再计算相似度
    if use_proj and proj is not None:
        image_features_proj, cache_keys_proj = proj(image_features, cache_keys)
        sim_weights = torch.matmul(image_features_proj, cache_keys_proj.to(device).t())  # b n
    else:
        sim_weights = ori_sim_weights

    # 如果启用 Faiss Top-K 稀疏注意力，仅保留 Top-K 个记忆项的相似度
    if use_faiss:
        sim_weights = _faiss_topk_attention(image_features, cache_keys, faiss_topk, device)

    # 如果启用 mask 机制，过滤掉相似度过高的样本（防止过拟合）
    if need_mask:
        # 取原始相似度 95% 分位数作为阈值
        th = torch.quantile(ori_sim_weights, 0.95, dim=-1, keepdim=True)
        mask = ori_sim_weights > th    # default 0.9
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
#   新增参数：
#     use_faiss: 是否启用 Faiss Top-K 稀疏注意力
#     faiss_topk: Top-K 的 K 值（仅当 use_faiss=True 时生效）
# ============================================================
def compute_patch_socre(patch_features, cache_keys, cache_values, ori_sim_weights=None,
        device=None, proj=None, need_mask=False, patch_projection=False, gt_mask=None,
        anomaly_threshold=0.5, is_mradft=False, use_proj=True,
        use_faiss=False, faiss_topk=50):

    # 计算每个 patch 特征与记忆库 keys 的原始相似度
    ori_sim_weights = torch.matmul(patch_features, cache_keys.to(device).t())  # b 1369 n

    # 如果启用投影适配器，分别对 patch 特征和记忆库 keys 投影后再计算相似度
    if use_proj and proj is not None:
        patch_features_proj = proj(patch_features, 0)
        cache_keys_proj = proj(cache_keys, 1)
        sim_weights = torch.matmul(patch_features_proj, cache_keys_proj.T.to(device))  # b 1369 n
    else:
        sim_weights = ori_sim_weights

    # 如果启用 Faiss Top-K 稀疏注意力，对每个 patch 仅保留 Top-K 个记忆项
    if use_faiss:
        B, num_patches, D = patch_features.shape
        # 将 (B, num_patches, D) 展平为 (B*num_patches, D) 后调用 faiss
        sim_weights = _faiss_topk_attention(
            patch_features.reshape(-1, D), cache_keys, faiss_topk, device
        ).view(B, num_patches, -1)

    # 保存微调阶段的相似度副本
    finetune_sim_weights = sim_weights.clone()
    # 如果启用 mask 机制，过滤掉相似度过高的 patch（防止过拟合）
    if need_mask:
        # 取原始相似度 80% 分位数作为阈值
        th = torch.quantile(ori_sim_weights, 0.8, dim=-1, keepdim=True)
        mask = ori_sim_weights > th  # test mvtec     when test visa be setted  0.85 memclip be setted 0.95
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
    return logits, new_patch_features, ori_sim_weights, finetune_sim_weights
