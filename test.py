import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from models.prompt_learner import AnomalyCLIP_PromptLearner
from utils.loss import FocalLoss, BinaryDiceLoss
from utils.transforms import normalize, get_transform
from utils.dataset import Dataset
from utils.logger import get_logger
from tqdm import tqdm
from models.mlp import AnomalyMLP, MLP, Projector, average_neighbor
from models.attention import NormalFeatureAttention, CrossAttentionPooling
import os
import time
import random
import numpy as np
from tabulate import tabulate
from mrad import (
    build_cache_model, compute_socre, compute_patch_socre,
    build_patch_cache_model, CrossAttentionRetrieval, ScaleWeightedFusion
)
from utils.visualization import visualizer
from utils.metrics import image_level_metrics, pixel_level_metrics
from scipy.ndimage import gaussian_filter

import csv


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test(args):
    img_size = args.image_size
    features_list = args.features_list  # e.g. [6, 12, 18, 24]
    num_scales = len(features_list)
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    k = args.k
    if dataset_name == 'mvtec':
        seg_classi = 'mvtec'
        cache_name = 'visa'
    elif dataset_name == 'visa':
        cache_name = 'mvtec'
        seg_classi = 'visa'
    else:
        cache_name = 'visa'
        seg_classi = 'mvtec'

    logger = get_logger(args.save_path)
    device = args.device

    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx
    }

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device,
                                     design_details=AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess,
                        target_transform=target_transform, dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    obj_list = test_data.obj_list

    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['pixel-ap'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    image_proj = MLP()
    patch_proj = Projector(1024, 768, length=2)
    prompt_proj = AnomalyMLP()
    normal_atten = CrossAttentionPooling()

    # 新增：交叉注意力检索模块
    cross_attn = None
    if args.use_cross_attn:
        cross_attn = CrossAttentionRetrieval(
            embed_dim=768,
            num_heads=args.cross_attn_heads,
            dropout=0.1
        )
        cross_attn.to(device)

    # 新增：多尺度得分融合模块
    scale_fusion = ScaleWeightedFusion(num_scales=num_scales)
    scale_fusion.to(device)

    model_type = args.model_type

    # Only mrad-clip and mrad-ft need to load weights
    if model_type in ['mrad-clip', 'mrad-ft']:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        image_proj.load_state_dict(checkpoint["image_proj"])
        image_proj.to(device)
        patch_proj.load_state_dict(checkpoint["patch_proj"])
        patch_proj.to(device)
        # 加载多尺度融合模块权重
        if "scale_fusion" in checkpoint:
            scale_fusion.load_state_dict(checkpoint["scale_fusion"])
            scale_fusion.to(device)
            # 加载融合权重值
            if "scale_weights" in checkpoint:
                scale_fusion.scale_weights.data = checkpoint["scale_weights"].to(device)
        # 加载交叉注意力模块权重
        if cross_attn is not None and "cross_attn" in checkpoint:
            cross_attn.load_state_dict(checkpoint["cross_attn"])
            cross_attn.to(device)

    # Only mrad-clip needs prompt_learner and prompt_proj
    if model_type == 'mrad-clip':
        prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        prompt_learner.to(device)
        prompt_proj.load_state_dict(checkpoint["prompt_proj"])
        prompt_proj.to(device)

    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=24)

    # ================================================================
    # 构建图像级记忆库
    # ================================================================
    image_cache_path = os.path.join(args.cache_dir, f'cache_model_{cache_name}.pt')
    if not os.path.exists(image_cache_path):
        raise FileNotFoundError(
            f"图像级缓存文件不存在: {image_cache_path}\n"
            f"请先运行 train.py 构建记忆库缓存。"
        )
    cache_key, cache_value = build_cache_model(
        load_cache=True, clip_model=model, train_loader_cache=None,
        device=device, dir=image_cache_path
    )
    # 构建 patch 级记忆库（多尺度 + KMeans 聚类）
    patch_cache_path = os.path.join(args.cache_dir, f'cache_patch_model_{cache_name}.pt')
    if not os.path.exists(patch_cache_path):
        raise FileNotFoundError(
            f"Patch 级缓存文件不存在: {patch_cache_path}\n"
            f"请先运行 train.py 构建记忆库缓存。"
        )
    cache_keys_patch = build_patch_cache_model(
        load_cache=True, clip_model=model, train_loader_cache=None,
        device=device, dir=patch_cache_path,
        k_clusters=args.k_clusters if hasattr(args, 'k_clusters') else 8,
        use_kmeans=args.use_kmeans if hasattr(args, 'use_kmeans') else True,
        multi_scale=args.multi_scale,
        features_list=features_list
    )

    # 统一为 dict 格式
    if not args.multi_scale:
        ck, cv = cache_keys_patch
        cache_keys_patch = {3: {'keys': ck, 'values': cv}}

    print(f"cache_key (image): {cache_key.shape}")
    for layer_idx in range(num_scales):
        actual_idx = layer_idx if args.multi_scale else 3
        if actual_idx in cache_keys_patch:
            print(f"  cache_patch L{features_list[layer_idx]}: "
                  f"{cache_keys_patch[actual_idx]['keys'].shape[0]} prototypes")

    # ================================================================
    # 推理阶段：交叉注意力模块设为 eval 模式
    # ================================================================
    if cross_attn is not None:
        cross_attn.eval()

    start = time.time()
    norm_num, anom_num = 0, 0
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        if items['anomaly'] == 0:
            norm_num += 1
        else:
            anom_num += 1

        with torch.no_grad():
            image_features, patch_features, all_cls_tokens, patch_projections = \
                model.encode_image(image, features_list, DPAM_layer=24)

            # ============================================================
            # 多尺度 patch 特征处理
            # ============================================================
            patch_projections_list = []
            patch_features_list = []
            for layer_idx in range(num_scales):
                # 投影特征（768-dim）
                pp = patch_projections[layer_idx]
                pp = average_neighbor(pp)
                pp = pp / pp.norm(dim=-1, keepdim=True)
                patch_projections_list.append(pp)
                # patch 特征（1024-dim）
                pf = patch_features[layer_idx]
                pf = average_neighbor(pf)
                pf = pf / pf.norm(dim=-1, keepdim=True)
                patch_features_list.append(pf)

            # Decide whether to use projection
            use_proj = (model_type != 'mrad-tf')

            # ============================================================
            # 计算多尺度 patch 级分割 logits
            # ============================================================
            if args.multi_scale:
                seg_logit, patch_f_bia, ori_weight, ft_weight = compute_patch_socre(
                    patch_features_list, cache_keys_patch,
                    cache_values=None,
                    device=device,
                    proj=patch_proj if use_proj else None,
                    need_mask=False,
                    patch_projection=patch_projections_list,
                    gt_mask=items['img_mask'],
                    is_mradft=(model_type != 'mrad-clip'),
                    use_proj=use_proj,
                    cross_attn=cross_attn if use_proj else None,
                    use_faiss=args.use_faiss if hasattr(args, 'use_faiss') else False,
                    faiss_topk=args.faiss_topk if hasattr(args, 'faiss_topk') else 50,
                    scale_fusion=scale_fusion
                )
            else:
                seg_logit, patch_f_bia, ori_weight, ft_weight = compute_patch_socre(
                    patch_features_list[-1],
                    cache_keys_patch[3]['keys'],
                    cache_keys_patch[3]['values'],
                    device=device,
                    proj=patch_proj if use_proj else None,
                    need_mask=False,
                    patch_projection=patch_projections_list[-1],
                    gt_mask=items['img_mask'],
                    is_mradft=(model_type != 'mrad-clip'),
                    use_proj=use_proj,
                    cross_attn=cross_attn if use_proj else None,
                    use_faiss=args.use_faiss if hasattr(args, 'use_faiss') else False,
                    faiss_topk=args.faiss_topk if hasattr(args, 'faiss_topk') else 50,
                    scale_fusion=None
                )

            seg_similarity_map = AnomalyCLIP_lib.get_similarity_map(seg_logit, args.image_size)

            # CLIP image-text alignment (only needed for mrad-clip)
            if model_type == 'mrad-clip':
                bias = prompt_proj(patch_f_bia[:, 0, :], patch_f_bia[:, 1, :])
                prompts, tokenized_prompts, compound_prompts_text = \
                    prompt_learner(cls_id=None, bias=bias)
                text_features = model.encode_text_learn(
                    prompts, tokenized_prompts, compound_prompts_text
                ).float()
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cache_logits, _ = compute_socre(
                image_features, cache_key, cache_value, device,
                proj=image_proj if use_proj else None,
                use_proj=use_proj
            )

            text_probs = cache_logits
            text_probs = text_probs[:, 1]
            anomaly_map_list = []

            # ============================================================
            # 构建异常图列表
            # ============================================================
            for layer_idx, patch_feature in enumerate(patch_projections):
                # 仅处理指定层及之后的特征图
                if layer_idx >= args.feature_map_layer[3]:
                    patch_feature_proc = average_neighbor(patch_feature)
                    patch_feature_proc = patch_feature_proc / patch_feature_proc.norm(dim=-1, keepdim=True)

                    if model_type == 'mrad-clip':
                        # mrad-clip 使用 CLIP 图文对齐异常图
                        similarity = AnomalyCLIP_lib.compute_similarity(
                            patch_feature_proc, text_features
                        )
                        similarity_map = AnomalyCLIP_lib.get_similarity_map(
                            similarity, args.image_size
                        )
                        anomaly_map1 = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                        anomaly_map_list.append(anomaly_map1)
                    else:
                        # mrad-ft 和 mrad-tf 使用多尺度融合后的 seg_similarity_map
                        # 每层都使用同一个融合后的结果
                        anomaly_map2 = seg_similarity_map[..., 1]
                        anomaly_map_list.append(anomaly_map2)

            # 高斯滤波平滑 + 求和融合
            anomaly_map = torch.stack(anomaly_map_list)  # [num_layers, 1, 518, 518]
            anomaly_map = torch.stack([
                torch.from_numpy(gaussian_filter(i, sigma=args.sigma))
                for i in anomaly_map.detach().cpu()
            ], dim=0)

            anomaly_map = anomaly_map.sum(dim=0)
            map_flat = anomaly_map.flatten()
            topk_values, _ = torch.topk(map_flat, int(map_flat.shape[0] * 0.01))
            topk_mean = topk_values.mean()
            text_probs = (1 - k) * topk_mean + k * text_probs
            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)

        if args.visulize_bool:
            visualizer(
                items['img_path'][0], gt_mask,
                anomaly_map.detach().cpu().numpy(),
                save_dir=os.path.join(args.save_path, 'visualization', dataset_name),
                img_size=args.image_size, data_dir=dataset_dir
            )

    end = time.time()
    print(f"time cost: {end - start}")

    # ================================================================
    # 指标计算与输出
    # ================================================================
    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    compute_pixel_aupro = args.compute_pixel_aupro
    pixel_aupro_list = [] if compute_pixel_aupro else None
    pixel_f1_list = []
    for obj in tqdm(obj_list):
        table = []
        table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
        elif args.metrics == 'pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            if compute_pixel_aupro:
                pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
                table.append(str(np.round(pixel_aupro * 100, decimals=1)))
                pixel_aupro_list.append(pixel_aupro)
        elif args.metrics == 'image-pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            if compute_pixel_aupro:
                pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
                table.append(str(np.round(pixel_aupro * 100, decimals=1)))
                pixel_aupro_list.append(pixel_aupro)
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_ap_list.append(image_ap)

        table_ls.append(table)

    # 根据 objects 的字母顺序对 table_ls 进行排序
    table_ls.sort(key=lambda x: x[0])

    if args.metrics == 'image-level':
        table_ls.append(['mean',
            str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
            str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
    elif args.metrics == 'pixel-level':
        mean_row = ['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1))]
        headers = ['objects', 'pixel_auroc']
        if compute_pixel_aupro and pixel_aupro_list:
            mean_row.append(str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)))
            headers.append('pixel_aupro')
        table_ls.append(mean_row)
        results = tabulate(table_ls, headers=headers, tablefmt="pipe")
    elif args.metrics == 'image-pixel-level':
        mean_row = ['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1))]
        headers = ['objects', 'pixel_auroc']
        if compute_pixel_aupro and pixel_aupro_list:
            mean_row.append(str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)))
            headers.append('pixel_aupro')
        mean_row.extend([
            str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
            str(np.round(np.mean(image_ap_list) * 100, decimals=1))
        ])
        headers.extend(['image_auroc', 'image_ap'])
        table_ls.append(mean_row)
        results = tabulate(table_ls, headers=headers, tablefmt="pipe")
    logger.info("\n%s", results)

    # 保存 results 到文本文件
    with open(os.path.join(args.save_path, f'log-{args.model_index}.txt'), 'w') as f:
        f.write(results)

    # 保存 table_ls 到 CSV 文件
    with open(os.path.join(args.save_path, f'log-{args.model_index}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        if args.metrics == 'image-level':
            writer.writerow(['objects', 'image_auroc', 'image_ap'])
        elif args.metrics == 'pixel-level':
            headers = ['objects', 'pixel_auroc']
            if compute_pixel_aupro and pixel_aupro_list:
                headers.append('pixel_aupro')
            writer.writerow(headers)
        elif args.metrics == 'image-pixel-level':
            headers = ['objects', 'pixel_auroc']
            if compute_pixel_aupro and pixel_aupro_list:
                headers.append('pixel_aupro')
            headers.extend(['image_auroc', 'image_ap'])
            writer.writerow(headers)
        for row in table_ls:
            writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MRAD Testing", add_help=True)
    # Paths
    parser.add_argument("--data_path", type=str,
        default="/home/ts-cjh/Data/MRAD/data/mvtec_anomaly_detection",
        help="path to test dataset")
    parser.add_argument("--save_path", type=str,
        default='./results/test_on_mvtec', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str,
        default='./checkpoints/released/mrad_clip_final.pth',
        help='path to checkpoint')
    parser.add_argument("--cache_dir", type=str, default='./cache',
        help='directory for cache files')
    # Model parameters
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+",
        default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518,
        help="image size")
    parser.add_argument("--depth", type=int, default=9,
        help="learnable_text_embedding_depth")
    parser.add_argument("--n_ctx", type=int, default=12,
        help="prompt length")
    parser.add_argument("--t_n_ctx", type=int, default=4,
        help="learnable_text_embedding_length")
    parser.add_argument("--feature_map_layer", type=int, nargs="+",
        default=[0, 1, 2, 3], help="feature map layers")
    parser.add_argument("--metrics", type=str, default='image-pixel-level',
        help='metrics: image-level, pixel-level, image-pixel-level')
    parser.add_argument("--seed", type=int, default=111,
        help="random seed")
    parser.add_argument("--sigma", type=int, default=4,
        help="gaussian filter sigma")
    parser.add_argument("--k", type=float, default=0.7,
        help="fusion weight (0.5, 0.7, 0.8)")
    parser.add_argument("--device", type=str, default='cuda:1')
    parser.add_argument("--visulize_bool", type=bool, default=False)
    parser.add_argument("--compute_pixel_aupro", type=bool, default=True,
        help="compute pixel-level AUPRO metric in addition to AUROC")
    parser.add_argument("--model_type", type=str, default='mrad-clip',
        choices=['mrad-clip', 'mrad-ft', 'mrad-tf'],
        help='Model type: mrad-clip (full), mrad-ft (fine-tuned), mrad-tf (train-free)')
    parser.add_argument("--model_index", type=int, default=0,
        help="model index for logging")

    # ================================================================
    # 新增：改进方案的控制参数（需与训练时保持一致）
    # ================================================================
    parser.add_argument("--multi_scale", action="store_true", default=True,
        help="使用多尺度记忆库")
    parser.add_argument("--no_multi_scale", action="store_false", dest="multi_scale",
        help="禁用多尺度记忆库")
    parser.add_argument("--use_kmeans", action="store_true", default=True,
        help="使用 KMeans 聚类记忆库")
    parser.add_argument("--no_kmeans", action="store_false", dest="use_kmeans",
        help="使用均值记忆库")
    parser.add_argument("--k_clusters", type=int, default=8,
        help="KMeans 聚类数量")
    parser.add_argument("--use_cross_attn", action="store_true", default=True,
        help="使用交叉注意力检索")
    parser.add_argument("--no_cross_attn", action="store_false", dest="use_cross_attn",
        help="禁用交叉注意力")
    parser.add_argument("--cross_attn_heads", type=int, default=8,
        help="交叉注意力头数")
    parser.add_argument("--use_faiss", action="store_true", default=False,
        help="使用 Faiss 稀疏检索")
    parser.add_argument("--faiss_topk", type=int, default=50,
        help="Faiss Top-K 数量")

    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
