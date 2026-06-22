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
# 导入 KMeans 聚类算法（用于记忆库构建中的聚类操作）
from sklearn.cluster import KMeans
# 导入 os 模块，用于多尺度 memory bank 文件路径拼接
import os


# ============================================================
# build_cache_model: 构建图像级记忆库（正常/异常分类的 keys 和 values）
# ============================================================
def build_cache_model(load_cache = False,  clip_model = None, train_loader_cache = None,device = None,dir=None):
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

        # 将构建好的记忆库保存到磁盘
        torch.save({"keys": cache_keys.cpu(), "values": cache_values.cpu()}, cache_dir)

    # --- 分支2: 从磁盘加载已有记忆库 ---
    else:
        cache_dict = torch.load(cache_dir, map_location="cpu")
        cache_keys = cache_dict["keys"].to(device)
        cache_values = cache_dict["values"].to(device)
    return cache_keys, cache_values


# ============================================================
# build_patch_cache_model: 构建 patch 级记忆库（逐样本收集正常/异常 patch 的均值特征）
# ============================================================
def build_patch_cache_model(load_cache = False,  clip_model = None, train_loader_cache = None,device = None,dir=None, multi_scale=False):
    cache_dir = dir
    # --- 分支1: 从头构建 patch 级记忆库 ---
    if load_cache == False:
        # ============================================================
        # 多尺度模式：为 layers 6, 12, 18, 24 分别构建独立 memory bank
        # ============================================================
        if multi_scale:
            num_layers = 4
            # 为每层独立维护特征和标签列表
            all_train_features = [[] for _ in range(num_layers)]
            all_train_labels = [[] for _ in range(num_layers)]

            # 禁用梯度计算
            with torch.no_grad():
                # 遍历训练数据，逐样本提取各层 patch 特征
                for items in tqdm(train_loader_cache):
                    images = items['img'].to(device)
                    labels = items['anomaly'].to(device)
                    gt = items['img_mask'].squeeze().to(device)
                    # 将 ground truth mask 二值化
                    gt[gt > 0.5] = 1
                    gt[gt <= 0.5] = 0
                    # 使用 CLIP 模型编码图像，一次性获取所有层的 patch 特征
                    image_fe, patch_features, _, patch_projections = clip_model.encode_image(
                        images, [6, 12, 18, 24], DPAM_layer=24
                    )
                    # 将 ground truth mask 下采样到 patch 特征图尺寸 (37x37)
                    gt_resized = F.interpolate(gt.unsqueeze(1), size=(37, 37), mode='bilinear', align_corners=False)
                    gt_resized = gt_resized.squeeze(1)

                    # 逐层处理 patch 特征，每层独立收集正常/异常原型
                    for layer_idx in range(num_layers):
                        patch_feature = patch_features[layer_idx]
                        # 对 patch 特征进行邻域平均
                        patch_feature = average_neighbor(patch_feature)

                        # 逐样本按 mask 分类 patch 为异常和正常
                        for i in range(images.size(0)):
                            patch = patch_feature[i]              # (1369, 768)
                            patch = patch.view(37, 37, -1)        # (37, 37, 768)
                            mask = gt_resized[i]                  # (37, 37)

                            pos_mask = (mask == 1)                # 异常区域
                            neg_mask = (mask == 0)                # 正常区域

                            # 异常 patch 均值特征加入记忆库（标签=1）
                            if pos_mask.sum() > 0:
                                pos_feat = patch[pos_mask]
                                pos_feat = pos_feat.mean(dim=0, keepdim=True)
                                pos_feat = pos_feat / pos_feat.norm(dim=-1, keepdim=True)
                                all_train_features[layer_idx].append(pos_feat)
                                all_train_labels[layer_idx].append(torch.tensor([1], device=device))

                            # 正常 patch 均值特征加入记忆库（标签=0）
                            if neg_mask.sum() > 0:
                                neg_feat = patch[neg_mask]
                                neg_feat = neg_feat.mean(dim=0, keepdim=True)
                                neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)
                                all_train_features[layer_idx].append(neg_feat)
                                all_train_labels[layer_idx].append(torch.tensor([0], device=device))

            # 逐层拼接特征、生成 one-hot 标签并保存到独立文件
            cache_keys_list = []
            cache_values_list = []
            for layer_idx in range(num_layers):
                ck = torch.cat(all_train_features[layer_idx], dim=0)
                rl = torch.cat(all_train_labels[layer_idx], dim=0).to(torch.int64)
                cv = F.one_hot(rl, num_classes=2).float().to(device)
                cache_keys_list.append(ck)
                cache_values_list.append(cv)
                # 在文件名中插入层索引后缀（如 cache_patch_model_visa_l0.pt）
                base, ext = os.path.splitext(cache_dir)
                layer_dir = f"{base}_l{layer_idx}{ext}"
                torch.save({"keys": ck.cpu(), "values": cv.cpu()}, layer_dir)

            return cache_keys_list, cache_values_list

        # ============================================================
        # 原始单层模式（multi_scale=False，保持向后兼容）
        # ============================================================
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

        # 保存 patch 级记忆库到磁盘
        torch.save({"keys": cache_keys.cpu(), "values": cache_values.cpu()}, cache_dir)

    # --- 分支2: 从磁盘加载已有 patch 级记忆库 ---
    else:
        # ============================================================
        # 多尺度模式：按层索引分别加载 4 个 memory bank 文件
        # ============================================================
        if multi_scale:
            cache_keys_list = []
            cache_values_list = []
            for layer_idx in range(4):
                base, ext = os.path.splitext(cache_dir)
                layer_dir = f"{base}_l{layer_idx}{ext}"
                cache_dict = torch.load(layer_dir, map_location="cpu")
                cache_keys_list.append(cache_dict["keys"].to(device))
                cache_values_list.append(cache_dict["values"].to(device))
            return cache_keys_list, cache_values_list

        # 原始单层模式：加载单个文件
        cache_dict = torch.load(cache_dir, map_location="cpu")
        cache_keys = cache_dict["keys"].to(device)
        cache_values = cache_dict["values"].to(device)
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
        anomaly_threshold=0.5, is_mradft=False, use_proj=True,
        multi_scale=False, scale_weights=None):

    # ============================================================
    # 多尺度模式：逐层检索（layers 6/12/18/24）后加权融合 logits
    # ============================================================
    if multi_scale:
        # 对可学习融合权重做 softmax 归一化，确保权重和为 1
        normalized_weights = F.softmax(scale_weights, dim=0)  # [4]
        num_layers = len(patch_features)  # 4

        logits_list = []
        last_ori_sim = None
        last_ft_sim = None

        # 逐层计算相似度检索和 logits
        for layer_idx in range(num_layers):
            pf = patch_features[layer_idx]  # (B, 1369, 768)
            ck = cache_keys[layer_idx]      # (N_l, 768)
            cv = cache_values[layer_idx]    # (N_l, 2)

            # 计算当前层 patch 特征与记忆库 keys 的原始相似度
            ori_sim = torch.matmul(pf, ck.to(device).t())  # (B, 1369, N_l)

            # 如果启用投影适配器，分别投影 patch 特征和 memory keys
            if use_proj and proj is not None:
                pf_proj = proj(pf, 0)
                ck_proj = proj(ck, 1)
                sim = torch.matmul(pf_proj, ck_proj.T.to(device))
            else:
                sim = ori_sim

            # 保存微调阶段的相似度副本
            ft_sim = sim.clone()

            # mask 机制：过滤相似度过高的 patch（防过拟合）
            if need_mask:
                th = torch.quantile(ori_sim, 0.8, dim=-1, keepdim=True)
                mask = ori_sim > th
                sim = sim.masked_fill(mask, float('-inf'))

            # softmax 归一化并加权求和得到当前层 logits
            sim = F.softmax(sim, dim=-1)
            logits = torch.matmul(sim, cv.to(device).float())  # (B, 1369, 2)
            logits_list.append(logits)

            # 保留最后一层的原始相似度和微调相似度用于返回
            if layer_idx == num_layers - 1:
                last_ori_sim = ori_sim
                last_ft_sim = ft_sim

        # 多尺度加权融合：fused_logits = Σ(w_i * logits_i)
        fused_logits = sum(normalized_weights[i] * logits_list[i] for i in range(num_layers))

        # 使用融合后的 logits 计算 new_patch_features（非 mradft 模式）
        if not is_mradft:
            anomaly_probs = fused_logits[:, :, 1]
            anomaly_threshold_t = torch.tensor(anomaly_threshold, device=anomaly_probs.device, dtype=anomaly_probs.dtype)
            anomaly_area = (anomaly_probs > anomaly_threshold_t).float()  # (B, 1369)
            normal_area = 1.0 - anomaly_area

            anomaly_mask = anomaly_area.unsqueeze(-1)  # (B, 1369, 1)
            normal_mask = normal_area.unsqueeze(-1)

            # patch_projection 为最后一层的投影特征，用于计算 token 特征
            anomaly_feat_sum = (patch_projection * anomaly_mask).sum(dim=1)  # (B, 768)
            normal_feat_sum = (patch_projection * normal_mask).sum(dim=1)

            anomaly_count = anomaly_mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
            normal_count = normal_mask.sum(dim=1).clamp(min=1.0)

            anomaly_token = anomaly_feat_sum / anomaly_count  # (B, 768)
            normal_token = normal_feat_sum / normal_count

            anomaly_token[anomaly_mask.sum(dim=1).squeeze(-1) == 0] = 0.0
            normal_token[normal_mask.sum(dim=1).squeeze(-1) == 0] = 0.0

            new_patch_features = torch.stack([normal_token, anomaly_token], dim=1)  # (B, 2, 768)
        else:
            new_patch_features = 0

        return fused_logits, new_patch_features, last_ori_sim, last_ft_sim

    # ============================================================
    # 原始单层模式（multi_scale=False，保持向后兼容）
    # ============================================================
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
