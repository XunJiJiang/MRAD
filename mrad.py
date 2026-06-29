"""
MRAD — Multi-scale Cluster Memory Retrieval Anomaly Detection
================================================================
改进版本核心功能：
  1. KMeans 聚类记忆库 — 替代简单均值原型，保留多种缺陷模式
  2. 多尺度特征记忆 — 利用 L6/L12/L18/L24 四层特征建库
  3. 交叉注意力检索 — 使用 nn.MultiheadAttention 替代 softmax(QK^T)
  4. Faiss Top-K 稀疏注意力 — 可选，加速大规模记忆库检索

论文创新点：
  - Multi-scale Memory Retrieval
  - Cluster-aware Memory Retrieval
  - Cross-Attention Retrieval Fusion
  - Efficient Faiss-based Sparse Retrieval
"""

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from models.mlp import average_neighbor
import numpy as np

# 导入 KMeans 聚类算法
from sklearn.cluster import KMeans

# Faiss 可选导入（加速大规模检索）
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ============================================================
# CrossAttentionRetrieval: 交叉注意力检索模块
# 使用 MultiheadAttention 替代简单的 softmax(QK^T)，增强查询与记忆库之间的特征交互
# ============================================================
class CrossAttentionRetrieval(nn.Module):
    """交叉注意力检索模块

    将查询特征（patch features）与记忆库特征进行交叉注意力交互，
    使查询能够自适应地关注记忆库中最相关的原型。
    结构: MultiheadAttention → Add&Norm → FFN → Add&Norm

    Args:
        embed_dim: 特征维度（默认 768，与投影后维度一致）
        num_heads: 注意力头数
        dropout: dropout 概率
    """
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        # 多头交叉注意力层：Q=查询特征, K=V=记忆库特征
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # 第一个 Add&Norm：注意力残差连接后的层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        # 前馈网络：维度扩展 4 倍 → GELU 激活 → 压缩回原维度
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        # 第二个 Add&Norm：FFN 残差连接后的层归一化
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        """前向传播

        Args:
            query: [B, N_q, D] — 待增强的查询特征（如 patch 特征序列）
            key_value: [B, N_kv, D] — 记忆库特征（同时作为 Key 和 Value）
        Returns:
            [B, N_q, D] — 经过交叉注意力增强后的查询特征
        """
        # 交叉注意力：query 作为 Q，记忆库同时作为 K 和 V
        attn_output, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False
        )
        # 残差连接 + 层归一化
        query = self.norm1(query + attn_output)
        # 前馈网络 + 残差连接 + 层归一化
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)
        return query


# ============================================================
# ScaleWeightedFusion: 多尺度得分融合模块
# 为每个特征层学习一个可微的融合权重，通过 softmax 确保权重和为 1
# ============================================================
class ScaleWeightedFusion(nn.Module):
    """多尺度得分融合模块

    为每个特征层学习独立权重，在 log 空间存储并通过 softmax 归一化，
    确保各尺度权重非负且和为 1，便于梯度传播和解释各层贡献。

    Args:
        num_scales: 特征层数量（默认 4: L6, L12, L18, L24）
    """
    def __init__(self, num_scales=4):
        super().__init__()
        # 可学习的尺度权重（log 空间，初始化为 0 即均匀权重）
        self.scale_weights = nn.Parameter(torch.zeros(num_scales))
        # 存储最近一次前向的权重值（用于日志和可视化）
        self._last_weights = None

    def forward(self, scale_logits_list):
        """前向传播

        Args:
            scale_logits_list: list of [B, N, 2] — 各层的 patch 分类 logits
        Returns:
            [B, N, 2] — 加权融合后的 logits
        """
        # softmax 归一化权重：softmax([w1, w2, w3, w4]) → sum = 1
        weights = F.softmax(self.scale_weights, dim=0)  # [num_scales]
        # 加权求和各层 logits
        fused = sum(w * logit for w, logit in zip(weights, scale_logits_list))
        # 保存当前权重供外部访问
        self._last_weights = weights.detach().clone()
        return fused


# ============================================================
# _kmeans_cluster: KMeans 聚类辅助函数
# 对收集到的特征进行聚类，返回 L2 归一化的聚类中心作为原型
# ============================================================
def _kmeans_cluster(features, k, random_state=42, max_samples=50000):
    """KMeans 聚类并返回 L2 归一化的聚类中心

    Args:
        features: [N, D] numpy array — 待聚类的特征向量
        k: int — 目标聚类数量
        random_state: int — 随机种子（保证实验可复现）
        max_samples: int — 最大采样数，超出时随机下采样防止内存溢出
    Returns:
        [k_actual, D] tensor — L2 归一化后的聚类中心
    """
    N = features.shape[0]
    # 样本不足 k 个时，使用全部样本
    if N < k:
        k = N
    # 样本过多时随机下采样，防止 sklearn KMeans 内存溢出
    if N > max_samples:
        idx = np.random.choice(N, max_samples, replace=False)
        features = features[idx]

    # KMeans 聚类（在 L2 归一化后的特征上进行，等价于 spherical KMeans）
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(features)
    # 获取聚类中心并转为 tensor
    centers = kmeans.cluster_centers_
    centers = torch.from_numpy(centers).float()
    # L2 归一化聚类中心（确保与基于余弦相似度的检索兼容）
    centers = centers / centers.norm(dim=-1, keepdim=True)
    return centers


# ============================================================
# _faiss_topk_attention: Faiss Top-K 稀疏注意力检索
# CPU faiss IndexFlatIP 检索定位 Top-K 索引 → PyTorch einsum 回算精确内积保留梯度
# ============================================================
def _faiss_topk_attention(query, memory_keys, memory_values, topk, device):
    """Faiss Top-K 稀疏注意力检索

    使用 CPU faiss IndexFlatIP 快速定位与 query 最相似的 topk 个记忆库条目，
    然后仅对这些条目用 PyTorch einsum 计算精确 attention（保留梯度用于训练）。

    Args:
        query: [B*N, D] 或 [B, N, D] — 查询特征
        memory_keys: [M, D] — 记忆库键值
        memory_values: [M, 2] — 记忆库标签（one-hot 编码）
        topk: int — 每查询保留的 Top-K 数量
        device: torch.device — 计算设备
    Returns:
        logits: [B*N, 2] — 稀疏注意力加权后的分类 logits
    """
    if not FAISS_AVAILABLE:
        raise ImportError(
            "Faiss 未安装。请运行: pip install faiss-cpu\n"
            "或设置 use_faiss=False 使用标准密集检索。"
        )

    # 处理输入形状：统一展平为 [total_queries, D]
    if query.dim() == 3:
        B, N, D = query.shape
        query_flat = query.reshape(B * N, D)
    else:
        query_flat = query  # [B*N, D]
        B_times_N, D = query_flat.shape

    M = memory_keys.shape[0]
    # 确保 topk 不超过记忆库大小
    topk = min(topk, M)

    # 将 memory_keys 转为 CPU numpy 构建 faiss 内积索引
    mk_np = memory_keys.detach().cpu().numpy().astype(np.float32)
    # IndexFlatIP: 内积索引，对 L2 归一化向量等价于余弦相似度
    index = faiss.IndexFlatIP(D)
    index.add(mk_np)

    # 批量检索 Top-K
    q_np = query_flat.detach().cpu().numpy().astype(np.float32)
    _, topk_indices = index.search(q_np, topk)  # [total_queries, topk]
    topk_indices = torch.from_numpy(topk_indices).long().to(device)

    # 收集 Top-K keys 和 values
    mk_gpu = memory_keys.to(device)
    mv_gpu = memory_values.to(device)
    topk_keys = mk_gpu[topk_indices]       # [total_queries, topk, D]
    topk_values = mv_gpu[topk_indices]     # [total_queries, topk, 2]

    # PyTorch einsum 精确计算内积（保留梯度链）
    sim = torch.einsum('qd,qkd->qk', query_flat, topk_keys)  # [total_queries, topk]
    # softmax 归一化注意力权重
    sim_weights = F.softmax(sim, dim=-1)  # [total_queries, topk]
    # 加权求和记忆库 values
    logits = torch.bmm(sim_weights.unsqueeze(1), topk_values).squeeze(1)  # [total_queries, 2]

    return logits


# ============================================================
# build_cache_model: 构建图像级记忆库（用于图像级异常分类）
# ============================================================
def build_cache_model(load_cache=False, clip_model=None, train_loader_cache=None,
                      device=None, dir=None):
    """构建图像级记忆库

    存储所有训练样本的图像级 CLIP 特征及其正常/异常标签。
    用于图像级的异常分类（整张图是否异常）。

    Args:
        load_cache: 是否从磁盘加载已有记忆库（True=加载, False=从头构建）
        clip_model: CLIP 模型实例
        train_loader_cache: 训练数据加载器
        device: 计算设备
        dir: 缓存文件路径
    Returns:
        cache_keys: [N_samples, D] — 图像级特征
        cache_values: [N_samples, 2] — one-hot 标签 [正常, 异常]
    """
    cache_dir = dir
    # --- 从头构建记忆库 ---
    if load_cache == False:
        cache_keys = []
        cache_values = []
        # 禁用梯度计算，加速推理并节省显存
        with torch.no_grad():
            train_features = []
            train_labels = []
            # 遍历训练数据加载器，逐批提取图像特征
            for items in tqdm(train_loader_cache, desc="Building image cache"):
                images = items['img'].to(device)
                labels = items['anomaly'].to(device)
                # 使用 CLIP 模型编码图像，获取图像级特征
                image_features, _, _, _ = clip_model.encode_image(
                    images, [6, 12, 18, 24], DPAM_layer=24
                )
                # L2 归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                train_features.append(image_features)
                train_labels.append(labels)
            # 拼接所有 batch 的特征和标签
            cache_keys = torch.cat(train_features, dim=0)
            raw_labels = torch.cat(train_labels, dim=0).to(torch.int64)
            # one-hot 编码标签（2 类：正常/异常）
            cache_values = F.one_hot(raw_labels, num_classes=2).float().to(device)
        cache_dict = {"keys": cache_keys, "values": cache_values}
        # 保存到磁盘
        torch.save({"keys": cache_keys.cpu(), "values": cache_values.cpu()}, cache_dir)
    # --- 从磁盘加载已有记忆库 ---
    else:
        cache_dict = torch.load(cache_dir, map_location="cpu")
        cache_keys = cache_dict["keys"].to(device)
        cache_values = cache_dict["values"].to(device)
    return cache_keys, cache_values


# ============================================================
# _build_patch_cache_single: 为单个特征层构建 patch 级记忆库（内部函数）
# 支持 KMeans 聚类和均值两种构建方式
# ============================================================
def _build_patch_cache_single(clip_model, train_loader_cache, device,
                               layer_idx, k_clusters, use_kmeans):
    """为单个特征层构建 patch 级记忆库

    Args:
        clip_model: CLIP 模型实例
        train_loader_cache: 训练数据加载器
        device: 计算设备
        layer_idx: 特征层索引（0=L6, 1=L12, 2=L18, 3=L24）
        k_clusters: KMeans 聚类数量（use_kmeans=True 时生效）
        use_kmeans: 是否使用 KMeans 聚类（False 则用原始逐样本均值法）
    Returns:
        cache_keys: [M, D] — 记忆库键值（原型特征）
        cache_values: [M, 2] — one-hot 标签
    """
    # --- KMeans 聚类方式：收集全部正常/异常 patch → 分别聚类 → k 个原型 ---
    if use_kmeans:
        # 分别收集正常和异常 patch 特征（立即转 CPU numpy 避免 GPU OOM）
        norm_features_cpu = []
        anom_features_cpu = []
        with torch.no_grad():
            for items in tqdm(train_loader_cache, desc=f"Collecting L{6+6*layer_idx} patches"):
                images = items['img'].to(device)
                gt = items['img_mask'].squeeze().to(device)
                # GT mask 二值化：异常区域=1，正常区域=0
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                # 提取指定层的 patch 特征
                _, patch_features, _, _ = clip_model.encode_image(
                    images, [6, 12, 18, 24], DPAM_layer=24
                )
                patch_feat = patch_features[layer_idx]  # [B, 1369, 1024]
                # 邻域平均增强局部上下文信息
                patch_feat = average_neighbor(patch_feat)
                # L2 归一化（使后续 KMeans 等价于 spherical KMeans）
                patch_feat = patch_feat / patch_feat.norm(dim=-1, keepdim=True)
                # 下采样 GT mask 到 patch 特征图尺寸 (37x37)
                gt_resized = F.interpolate(
                    gt.unsqueeze(1), size=(37, 37),
                    mode='bilinear', align_corners=False
                )
                gt_resized = gt_resized.squeeze(1)  # [B, 37, 37]
                # 逐样本收集正常/异常 patch 特征
                for i in range(images.size(0)):
                    patch = patch_feat[i].view(37, 37, -1)  # [37, 37, 1024]
                    mask = gt_resized[i]  # [37, 37]
                    pos_mask = (mask == 1)  # 异常区域
                    neg_mask = (mask == 0)  # 正常区域
                    # 收集异常 patch 特征 → CPU numpy
                    if pos_mask.sum() > 0:
                        pos_feat = patch[pos_mask]  # [n_pos, 1024]
                        anom_features_cpu.append(pos_feat.cpu().numpy())
                    # 收集正常 patch 特征 → CPU numpy
                    if neg_mask.sum() > 0:
                        neg_feat = patch[neg_mask]  # [n_neg, 1024]
                        norm_features_cpu.append(neg_feat.cpu().numpy())

        # KMeans 聚类：正常 patch → k 个原型，异常 patch → k 个原型
        train_features = []
        train_labels = []
        # 正常 patch 聚类 → k 个正常原型（标签=0）
        if len(norm_features_cpu) > 0:
            norm_all = np.concatenate(norm_features_cpu, axis=0)
            norm_centers = _kmeans_cluster(norm_all, k_clusters)
            for c in norm_centers:
                train_features.append(c.unsqueeze(0))
                train_labels.append(torch.tensor([0], device=device))
        # 异常 patch 聚类 → k 个异常原型（标签=1）
        if len(anom_features_cpu) > 0:
            anom_all = np.concatenate(anom_features_cpu, axis=0)
            anom_centers = _kmeans_cluster(anom_all, k_clusters)
            for c in anom_centers:
                train_features.append(c.unsqueeze(0))
                train_labels.append(torch.tensor([1], device=device))
        # 拼接所有原型 → 记忆库
        cache_keys = torch.cat(train_features, dim=0).to(device)  # [2k, 1024]
        raw_labels = torch.cat(train_labels, dim=0).to(torch.int64)
        cache_values = F.one_hot(raw_labels, num_classes=2).float().to(device)  # [2k, 2]
    # --- 原始均值法：逐样本计算正常/异常 patch 均值作为原型（向后兼容） ---
    else:
        train_features = []
        train_labels = []
        with torch.no_grad():
            for items in tqdm(train_loader_cache, desc=f"Building L{6+6*layer_idx} cache (mean)"):
                images = items['img'].to(device)
                gt = items['img_mask'].squeeze().to(device)
                # GT mask 二值化
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                # 提取指定层的 patch 特征
                _, patch_features, _, _ = clip_model.encode_image(
                    images, [6, 12, 18, 24], DPAM_layer=24
                )
                patch_feat = patch_features[layer_idx]  # [B, 1369, 1024]
                # 邻域平均增强局部上下文
                patch_feat = average_neighbor(patch_feat)
                # 下采样 GT mask
                gt_resized = F.interpolate(
                    gt.unsqueeze(1), size=(37, 37),
                    mode='bilinear', align_corners=False
                )
                gt_resized = gt_resized.squeeze(1)
                # 逐样本：计算正常 patch 均值和异常 patch 均值
                for i in range(images.size(0)):
                    patch = patch_feat[i].view(37, 37, -1)  # [37, 37, 1024]
                    mask = gt_resized[i]
                    pos_mask = (mask == 1)  # 异常
                    neg_mask = (mask == 0)  # 正常
                    # 异常 patch 均值 → 标签 1
                    if pos_mask.sum() > 0:
                        pos_feat = patch[pos_mask]  # [n_pos, 1024]
                        pos_feat = pos_feat.mean(dim=0, keepdim=True)
                        pos_feat = pos_feat / pos_feat.norm(dim=-1, keepdim=True)
                        train_features.append(pos_feat)
                        train_labels.append(torch.tensor([1], device=device))
                    # 正常 patch 均值 → 标签 0
                    if neg_mask.sum() > 0:
                        neg_feat = patch[neg_mask]  # [n_neg, 1024]
                        neg_feat = neg_feat.mean(dim=0, keepdim=True)
                        neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)
                        train_features.append(neg_feat)
                        train_labels.append(torch.tensor([0], device=device))
        # 拼接所有原型
        cache_keys = torch.cat(train_features, dim=0)  # [2*samples, 1024]
        raw_labels = torch.cat(train_labels, dim=0).to(torch.int64)
        cache_values = F.one_hot(raw_labels, num_classes=2).float().to(device)

    return cache_keys, cache_values


# ============================================================
# build_patch_cache_model: 构建 patch 级记忆库（统一入口）
# 支持单层和多尺度两种模式，支持 KMeans 聚类和均值两种构建方式
# ============================================================
def build_patch_cache_model(load_cache=False, clip_model=None, train_loader_cache=None,
                            device=None, dir=None, k_clusters=8, use_kmeans=True,
                            multi_scale=True, features_list=None):
    """构建 patch 级记忆库

    Args:
        load_cache: 是否从磁盘加载已有记忆库
        clip_model: CLIP 模型实例
        train_loader_cache: 训练数据加载器
        device: 计算设备
        dir: 缓存文件路径
        k_clusters: KMeans 聚类数量（use_kmeans=True 时生效，默认 8）
        use_kmeans: 是否使用 KMeans 聚类（True=聚类原型, False=逐样本均值）
        multi_scale: 是否构建多尺度记忆库（True=所有层, False=仅 L24）
        features_list: 特征层列表，默认 [6, 12, 18, 24]
    Returns:
        多尺度模式: dict {layer_idx: {'keys': tensor, 'values': tensor}}
        单层模式: (cache_keys, cache_values) 元组
    """
    if features_list is None:
        features_list = [6, 12, 18, 24]
    cache_dir = dir

    # --- 从磁盘加载已有记忆库 ---
    if load_cache:
        cache_data = torch.load(cache_dir, map_location="cpu")
        if multi_scale:
            # 多尺度模式：将各层数据迁移到 device
            multi_scale_cache = {}
            for layer_idx_str, layer_data in cache_data.items():
                layer_idx = int(layer_idx_str)
                multi_scale_cache[layer_idx] = {
                    'keys': layer_data['keys'].to(device),
                    'values': layer_data['values'].to(device)
                }
            return multi_scale_cache
        else:
            # 单层模式：兼容原有格式
            cache_keys = cache_data["keys"].to(device)
            cache_values = cache_data["values"].to(device)
            return cache_keys, cache_values

    # --- 从头构建记忆库 ---
    if multi_scale:
        # 多尺度模式：为每个特征层独立构建记忆库
        multi_scale_cache = {}
        num_layers = len(features_list)
        for layer_idx in range(num_layers):
            keys, values = _build_patch_cache_single(
                clip_model, train_loader_cache, device,
                layer_idx=layer_idx,
                k_clusters=k_clusters,
                use_kmeans=use_kmeans
            )
            # 存储时转 CPU 以节省 GPU 显存
            multi_scale_cache[layer_idx] = {
                'keys': keys.cpu(),
                'values': values.cpu()
            }
            print(f"  Layer {features_list[layer_idx]}: "
                  f"norm_centers={keys.shape[0]//2 if use_kmeans else 'N/A'}, "
                  f"total_keys={keys.shape[0]}")
        # 保存到磁盘（CPU tensor）
        torch.save(multi_scale_cache, cache_dir)
        # 再迁移到 device 供后续使用
        for layer_idx in multi_scale_cache:
            multi_scale_cache[layer_idx]['keys'] = multi_scale_cache[layer_idx]['keys'].to(device)
            multi_scale_cache[layer_idx]['values'] = multi_scale_cache[layer_idx]['values'].to(device)
        return multi_scale_cache
    else:
        # 单层模式：仅构建最后一层（L24, layer_idx=3）的记忆库
        keys, values = _build_patch_cache_single(
            clip_model, train_loader_cache, device,
            layer_idx=3,
            k_clusters=k_clusters,
            use_kmeans=use_kmeans
        )
        print(f"  Single-scale (L24): total_keys={keys.shape[0]}")
        # 保存到磁盘
        torch.save({"keys": keys.cpu(), "values": values.cpu()}, cache_dir)
        return keys, values


# ============================================================
# compute_socre: 图像级分类评分（基于记忆库的相似度检索）
# ============================================================
def compute_socre(image_features, cache_keys, cache_values, device,
                  proj=None, need_mask=False, is_train=False, use_proj=True):
    """图像级异常分类评分

    计算图像特征与记忆库的相似度，通过 softmax 加权得到正常/异常分类 logits。

    Args:
        image_features: [B, D] — 图像级特征
        cache_keys: [M, D] — 记忆库键值
        cache_values: [M, 2] — 记忆库标签
        device: 计算设备
        proj: 投影适配器（MLP 模块或 None）
        need_mask: 是否使用 mask 过滤高相似度样本（防止过拟合）
        is_train: 是否处于训练模式（当前未使用）
        use_proj: 是否使用投影适配器
    Returns:
        logits: [B, 2] — 分类 logits（正常/异常）
        loss_keys: scalar tensor — 辅助损失项（当前为 0）
    """
    # 计算图像特征与记忆库 keys 的原始余弦相似度
    ori_sim_weights = torch.matmul(image_features, cache_keys.to(device).t())  # [B, M]
    loss_keys = torch.tensor(0.0, device=device)

    # 如果启用投影适配器，对特征投影后再计算相似度
    if use_proj and proj is not None:
        image_features_proj, cache_keys_proj = proj(image_features, cache_keys)
        sim_weights = torch.matmul(image_features_proj, cache_keys_proj.to(device).t())  # [B, M]
    else:
        sim_weights = ori_sim_weights

    # 如果启用 mask 机制，过滤相似度过高的样本（防止记忆库中的样本被直接复制）
    if need_mask:
        # 取原始相似度 95% 分位数作为阈值
        th = torch.quantile(ori_sim_weights, 0.95, dim=-1, keepdim=True)
        mask = ori_sim_weights > th
        # 将高于阈值的相似度置为 -inf，softmax 后权重趋近于 0
        sim_weights = sim_weights.masked_fill(mask, float('-inf'))
    # softmax 归一化相似度权重
    sim_weights = F.softmax(sim_weights, dim=-1)
    # 加权求和记忆库 values，得到分类 logits
    logits = torch.matmul(sim_weights, cache_values.to(device).float())
    return logits, loss_keys


# ============================================================
# _compute_patch_socre_single: 单层 patch 级分割评分（内部函数）
# 支持交叉注意力增强和 Faiss 稀疏检索两种可选机制
# ============================================================
def _compute_patch_socre_single(patch_features, cache_keys, cache_values,
        ori_sim_weights=None, device=None, proj=None, need_mask=False,
        patch_projection=None, gt_mask=None, anomaly_threshold=0.5,
        is_mradft=False, use_proj=True, cross_attn=None,
        use_faiss=False, faiss_topk=50):
    """单层 patch 级异常分割评分

    对每个 patch 计算与记忆库原型之间的相似度，得到正常/异常分类结果。

    Args:
        patch_features: [B, N, D] — patch 特征（已做 neighbor + L2 norm）
        cache_keys: [M, D] — 记忆库键值（原型特征）
        cache_values: [M, 2] — 记忆库标签（one-hot）
        ori_sim_weights: 预计算的原始相似度（可选）
        device: 计算设备
        proj: 投影适配器（Projector 模块）
        need_mask: 是否使用 mask 过滤
        patch_projection: [B, N, D_proj] — 已投影的 patch 特征（用于 token 提取）
        gt_mask: ground truth mask
        anomaly_threshold: 异常区域判定阈值
        is_mradft: 是否为 MRAD-FT 模式（True 时不计算 new_patch_features）
        use_proj: 是否使用投影适配器
        cross_attn: CrossAttentionRetrieval 模块（可选，用于增强查询特征）
        use_faiss: 是否使用 Faiss 稀疏检索
        faiss_topk: Faiss Top-K 数量
    Returns:
        logits: [B, N, 2] — patch 级分类 logits
        new_patch_features: [B, 2, D_proj] 或 0 — 正常/异常 token 特征
        ori_sim_weights: [B, N, M] — 原始相似度
        finetune_sim_weights: [B, N, M] — 微调后相似度
    """
    # 计算每个 patch 特征与记忆库 keys 的原始内积相似度
    ori_sim_weights = torch.matmul(patch_features, cache_keys.to(device).t())  # [B, N, M]

    # --- Faiss 稀疏检索路径：直接返回 logits，跳过后续 softmax 流程 ---
    if use_faiss and FAISS_AVAILABLE:
        logits = _faiss_topk_attention(
            patch_features, cache_keys, cache_values, faiss_topk, device
        )
        # 恢复形状（faiss 返回的是展平后的）
        B = patch_features.shape[0]
        N = patch_features.shape[1]
        logits = logits.reshape(B, N, 2)
        finetune_sim_weights = ori_sim_weights.clone()
        # Faiss 路径暂不支持 new_patch_features 计算
        new_patch_features = 0
        return logits, new_patch_features, ori_sim_weights, finetune_sim_weights

    # --- 标准检索路径 ---
    # 如果启用投影适配器，将特征投影到公共空间再计算相似度
    if use_proj and proj is not None:
        # 投影查询特征和记忆库键值
        patch_features_proj = proj(patch_features, 0)  # [B, N, 768]
        cache_keys_proj = proj(cache_keys, 1)            # [M, 768]

        # 如果启用交叉注意力，先用交叉注意力增强查询特征
        if cross_attn is not None:
            # 扩展记忆库到 batch 维度
            M_k = cache_keys_proj.shape[0]
            memory_expanded = cache_keys_proj.unsqueeze(0).expand(
                patch_features_proj.shape[0], M_k, -1
            )
            # 交叉注意力：查询特征与记忆库交互，增强特征表达
            patch_features_proj = cross_attn(patch_features_proj, memory_expanded)

        # 计算投影后的相似度矩阵
        sim_weights = torch.matmul(patch_features_proj, cache_keys_proj.T.to(device))  # [B, N, M]
    else:
        sim_weights = ori_sim_weights

    # 保存微调阶段的相似度副本（用于后续可能的分心或调试）
    finetune_sim_weights = sim_weights.clone()
    # 如果启用 mask 机制，过滤相似度过高的 patch（防止过拟合到特定记忆条目）
    if need_mask:
        # 取原始相似度 80% 分位数作为阈值
        th = torch.quantile(ori_sim_weights, 0.8, dim=-1, keepdim=True)
        mask = ori_sim_weights > th
        # 将高于阈值的相似度置为 -inf
        sim_weights = sim_weights.masked_fill(mask, float('-inf'))
    # softmax 归一化注意力权重
    sim_weights = F.softmax(sim_weights, dim=-1)
    # 加权求和得到每个 patch 的正常/异常 logits
    logits = torch.matmul(sim_weights, cache_values.to(device).float())  # [B, N, 2]

    # --- 计算异常/正常 token 特征（用于 prompt learner 的 bias 输入） ---
    if not is_mradft:
        # 提取异常概率（logits 中第 1 通道为异常类）
        anomaly_probs = logits[:, :, 1]
        anomaly_threshold = torch.tensor(
            anomaly_threshold, device=anomaly_probs.device, dtype=anomaly_probs.dtype
        )
        # 根据阈值划分异常区域和正常区域
        anomaly_area = (anomaly_probs > anomaly_threshold).float()  # [B, N]
        normal_area = 1.0 - anomaly_area
        anomaly_mask = anomaly_area.unsqueeze(-1)  # [B, N, 1]
        normal_mask = normal_area.unsqueeze(-1)

        # 分别计算异常区域和正常区域的 patch 投影特征之和
        anomaly_feat_sum = (patch_projection * anomaly_mask).sum(dim=1)  # [B, D_proj]
        normal_feat_sum = (patch_projection * normal_mask).sum(dim=1)

        # 统计异常/正常 patch 数量，clamp 防止除零
        anomaly_count = anomaly_mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        normal_count = normal_mask.sum(dim=1).clamp(min=1.0)

        # 计算异常和正常的平均 token 特征
        anomaly_token = anomaly_feat_sum / anomaly_count  # [B, D_proj]
        normal_token = normal_feat_sum / normal_count

        # 如果没有异常 patch，则将异常 token 置零
        anomaly_token[anomaly_mask.sum(dim=1).squeeze(-1) == 0] = 0.0
        # 如果没有正常 patch，则将正常 token 置零
        normal_token[normal_mask.sum(dim=1).squeeze(-1) == 0] = 0.0

        # 堆叠正常和异常 token → [B, 2, D_proj]
        new_patch_features = torch.stack([normal_token, anomaly_token], dim=1)
    else:
        # MRAD-FT 模式下不计算 new_patch_features
        new_patch_features = 0

    return logits, new_patch_features, ori_sim_weights, finetune_sim_weights


# ============================================================
# compute_patch_socre: patch 级评分（统一入口，支持单层和多尺度）
# 多尺度模式下自动融合各层 logits
# ============================================================
def compute_patch_socre(patch_features, cache_keys, cache_values, ori_sim_weights=None,
        device=None, proj=None, need_mask=False, patch_projection=None, gt_mask=None,
        anomaly_threshold=0.5, is_mradft=False, use_proj=True, cross_attn=None,
        use_faiss=False, faiss_topk=50, scale_fusion=None):
    """patch 级异常分割评分（统一入口）

    当 patch_features 为单个 tensor 时（单层模式），直接调用单层评分函数。
    当 patch_features 为 list 时（多尺度模式），分别对各层评分后通过
    scale_fusion 模块融合各层结果。

    Args:
        patch_features: [B, N, D] 或 list of [B, N, D]
        cache_keys: [M, D] 或 dict of {layer_idx: {'keys':..., 'values':...}}
        cache_values: [M, 2] 或 None（多尺度时从 cache_keys dict 中获取）
        scale_fusion: ScaleWeightedFusion 模块（多尺度融合时使用）
        其余参数同 _compute_patch_socre_single。
    Returns:
        logits: [B, N, 2] — 融合后的 patch 级分类 logits
        new_patch_features: [B, 2, D_proj] 或 0
        ori_sim_weights: 原始相似度（单层时返回，多层时返回 None）
        finetune_sim_weights: 微调后相似度（单层时返回，多层时返回 None）
    """
    # --- 多尺度模式：对每层独立评分后融合 ---
    if isinstance(patch_features, list) and isinstance(cache_keys, dict):
        layer_logits = []
        layer_pfbs = []
        # 对每一层独立计算 patch 评分
        for layer_idx, pf in enumerate(patch_features):
            layer_cache_keys = cache_keys[layer_idx]['keys']
            layer_cache_values = cache_keys[layer_idx]['values']
            # 多尺度时 patch_projection 也对应每层
            layer_pp = (
                patch_projection[layer_idx]
                if isinstance(patch_projection, list) else patch_projection
            )
            logit, pfb, _, _ = _compute_patch_socre_single(
                pf, layer_cache_keys, layer_cache_values,
                device=device, proj=proj, need_mask=need_mask,
                patch_projection=layer_pp,
                gt_mask=gt_mask, anomaly_threshold=anomaly_threshold,
                is_mradft=is_mradft, use_proj=use_proj,
                cross_attn=cross_attn,
                use_faiss=use_faiss, faiss_topk=faiss_topk
            )
            layer_logits.append(logit)
            layer_pfbs.append(pfb)
        # 多尺度融合
        if scale_fusion is not None:
            fused_logits = scale_fusion(layer_logits)
        else:
            # 无融合模块时使用简单平均
            fused_logits = torch.stack(layer_logits, dim=0).mean(dim=0)
        # 取最后一层的 new_patch_features（用于 prompt learner）
        new_patch_features = layer_pfbs[-1]
        # 融合权重已在 ScaleWeightedFusion.forward() 中自动保存至 _last_weights
        return fused_logits, new_patch_features, None, None

    # --- 单层模式（保持原有接口兼容） ---
    else:
        return _compute_patch_socre_single(
            patch_features, cache_keys, cache_values,
            ori_sim_weights=ori_sim_weights, device=device, proj=proj,
            need_mask=need_mask, patch_projection=patch_projection,
            gt_mask=gt_mask, anomaly_threshold=anomaly_threshold,
            is_mradft=is_mradft, use_proj=use_proj,
            cross_attn=cross_attn,
            use_faiss=use_faiss, faiss_topk=faiss_topk
        )
