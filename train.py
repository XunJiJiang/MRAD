# 导入核心依赖库
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
import numpy as np
import os
import random
from models.mlp import AnomalyMLP, MLP, Projector, average_neighbor
from mrad import build_cache_model, compute_socre, compute_patch_socre, build_patch_cache_model

# 固定随机种子，确保实验可复现
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 主训练函数
def train(args):
    # 初始化日志记录器
    logger = get_logger(args.save_path)
    # 获取数据预处理变换
    preprocess, target_transform = get_transform(args)
    # 选择训练设备（GPU 或 CPU）
    device = args.device if torch.cuda.is_available() else "cpu"
    model_type = args.model_type

    # 构建 AnomalyCLIP 模型的参数配置
    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx
    }

    # 加载预训练的 AnomalyCLIP 模型（ViT-L/14@336px）
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    # 构建训练数据集与数据加载器
    train_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    dataset_name = args.dataset
    cache_name = dataset_name

    # Initialize model components
    # 初始化 patch 投影器和图像投影器
    patch_proj = Projector(1024, 768, length=2)
    image_proj = MLP()
    patch_proj.to(device)
    image_proj.to(device)

    # mrad-clip requires additional components
    # mrad-clip 模式下需要额外的 prompt 相关组件
    if model_type == 'mrad-clip':
        # 初始化 prompt 投影器和 prompt 学习器
        prompt_proj = AnomalyMLP()
        prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
        prompt_learner.to(device)
        prompt_proj.to(device)

    # 将模型迁移到目标设备，并替换视觉编码器的 DAPM 层
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=24)

    # Loss functions
    # 实例化损失函数：Focal Loss 和 Dice Loss
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    # Load memory bank
    # 构建图像级记忆库（用于异常分类）
    cache_keys, cache_values = build_cache_model(
        load_cache=False, clip_model=model, train_loader_cache=train_dataloader,
        device=device, dir=os.path.join(args.cache_dir, f'cache_model_{dataset_name}.pt')
    )
    # 构建 patch 级记忆库（用于异常分割）
    cache_keys_patch, cache_values_patch = build_patch_cache_model(
        load_cache=False, clip_model=model, train_loader_cache=train_dataloader,
        device=device, dir=os.path.join(args.cache_dir, f'cache_patch_model_{dataset_name}.pt')
    )
    # 打印记忆库维度信息
    print(f"cache_key: {cache_keys.shape}")
    print(f"cache_keys_patch: {cache_keys_patch.shape}")

    # Set total epochs based on model_type
    # 根据模型类型确定总训练轮数
    if model_type == 'mrad-ft':
        total_epochs = args.ft_epochs  # default 1
    else:  # mrad-clip
        total_epochs = args.ft_epochs + args.clip_epochs  # default 1 + 5 = 6

    # Stage 1 optimizer: train patch_proj and image_proj (mrad-ft stage)
    # 第一阶段优化器：训练 patch_proj 和 image_proj
    optimizer_ft = torch.optim.Adam(
        list(patch_proj.parameters()) + list(image_proj.parameters()),
        lr=args.learning_rate, betas=(0.5, 0.999)
    )
    # Cosine learning rate decay only for standalone mrad-ft training
    # 仅在独立 mrad-ft 训练时使用余弦退火学习率调度
    if model_type == 'mrad-ft':
        ft_total_steps = args.ft_epochs * len(train_dataloader)
        scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_ft, T_max=ft_total_steps, eta_min=1e-6
        )

    # Stage 2 optimizer: train prompt_learner and prompt_proj (mrad-clip stage)
    # 第二阶段优化器：训练 prompt_learner 和 prompt_proj
    if model_type == 'mrad-clip':
        optimizer_clip = torch.optim.Adam(
            list(prompt_learner.parameters()) + list(prompt_proj.parameters()),
            lr=args.learning_rate, betas=(0.5, 0.999)
        )

    # 主训练循环，遍历所有 epoch
    for epoch in tqdm(range(total_epochs)):
        # 设置各模块的训练/评估模式
        model.eval()
        patch_proj.train()
        image_proj.train()

        # mrad-clip 模式下，prompt 相关模块也设为训练模式
        if model_type == 'mrad-clip':
            prompt_learner.train()
            prompt_proj.train()

        # 初始化各阶段损失记录列表
        loss_list = []
        seg_loss_list = []
        clip_loss_list = []

        # Determine current stage
        # 根据当前 epoch 判断所处训练阶段（FT 阶段或 CLIP 阶段）
        is_ft_stage = (epoch < args.ft_epochs)

        # 按 batch 遍历训练数据
        for items in tqdm(train_dataloader):
            # 将图像、标签和 GT mask 迁移到目标设备
            image = items['img'].to(device)
            label = items['anomaly']
            gt = items['img_mask'].squeeze().to(device)
            # 将 GT mask 二值化
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            # 在无梯度上下文中提取图像特征（冻结 CLIP 主干网络）
            with torch.no_grad():
                image_features, patch_features, all_cls_tokens, patch_projections = model.encode_image(
                    image, args.features_list, DPAM_layer=24
                )

            # 处理 patch 投影特征：近邻平均 + L2 归一化
            patch_pojection = patch_projections[3]
            patch_pojection = average_neighbor(patch_pojection)
            patch_pojection = patch_pojection / patch_pojection.norm(dim=-1, keepdim=True)

            # 处理 patch 特征：近邻平均 + L2 归一化
            patch_f_bia = patch_features[3]
            patch_f_bia = average_neighbor(patch_f_bia)
            patch_f_bia = patch_f_bia / patch_f_bia.norm(dim=-1, keepdim=True)

            # Compute segmentation logits
            # 计算 patch 级的异常分割 logits
            seg_logit, patch_f_bia, _, _ = compute_patch_socre(
                patch_f_bia, cache_keys_patch, cache_values_patch,
                device=device, proj=patch_proj, need_mask=True,
                patch_projection=patch_pojection, use_proj=True
            )
            # 将分割 logits 转换为相似度图
            seg_similarity_map = AnomalyCLIP_lib.get_similarity_map(seg_logit, args.image_size).permute(0, 3, 1, 2)

            # Segmentation loss
            # 计算分割损失（Focal Loss + Dice Loss）
            seg_loss = loss_focal(seg_similarity_map, gt) + loss_dice(seg_similarity_map[:, 1, :, :], gt)
            seg_loss_list.append(seg_loss.item())

            # 根据当前阶段执行不同的训练逻辑
            if is_ft_stage:
                # Stage 1: train patch_proj (segmentation) and image_proj (classification)
                # 第一阶段：训练 patch_proj（分割）和 image_proj（分类）
                # Classification loss
                # 计算图像级分类损失
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits, _ = compute_socre(image_features, cache_keys, cache_values, device, proj=image_proj, need_mask=True, is_train=False)
                logits = logits / 0.07
                image_loss = F.cross_entropy(logits.squeeze(), label.long().to(device))

                # Total loss = segmentation loss + classification loss
                # 总损失 = 分割损失 + 分类损失
                total_loss = seg_loss + image_loss

                # 反向传播并更新第一阶段参数
                optimizer_ft.zero_grad()
                total_loss.backward()
                optimizer_ft.step()
                # 仅在独立 mrad-ft 模式下更新学习率调度器
                if model_type == 'mrad-ft':
                    scheduler_ft.step()
                loss_list.append(total_loss.item())
            else:
                # Stage 2: train prompt_learner and prompt_proj (mrad-clip)
                # 第二阶段：训练 prompt_learner 和 prompt_proj
                # Freeze patch_proj
                # 冻结 patch_proj 参数，不再更新
                for param in patch_proj.parameters():
                    param.requires_grad = False

                # 通过 prompt_proj 计算 bias，并由 prompt_learner 生成可学习 prompt
                bias = prompt_proj(patch_f_bia[:, 0, :], patch_f_bia[:, 1, :])
                prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None, bias=bias)
                # 使用可学习 prompt 编码文本特征，并进行 L2 归一化
                text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Compute CLIP alignment loss
                # 遍历各层 patch 投影特征，计算与文本特征的相似度图
                similarity_map_list = []
                for idx, patch_feature in enumerate(patch_projections):
                    # 仅处理指定层及之后的特征图
                    if idx >= args.feature_map_layer[3]:
                        # 近邻平均 + L2 归一化
                        patch_feature = average_neighbor(patch_feature)
                        patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                        # 计算 patch 特征与文本特征之间的相似度
                        similarity = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
                        # 将相似度转换为空间相似度图
                        similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity, args.image_size).permute(0, 3, 1, 2)
                        similarity_map_list.append(similarity_map)

                # 累加所有层相似度图的 CLIP 对齐损失
                clip_loss = 0
                for sim_map in similarity_map_list:
                    clip_loss += loss_focal(sim_map, gt)
                    clip_loss += loss_dice(sim_map[:, 1, :, :], gt)
                    clip_loss += loss_dice(sim_map[:, 0, :, :], 1 - gt)

                clip_loss_list.append(clip_loss.item())

                # 反向传播并更新第二阶段参数
                optimizer_clip.zero_grad()
                clip_loss.backward()
                optimizer_clip.step()
                loss_list.append(clip_loss.item())

        # Logging
        # 按指定频率输出训练日志
        if (epoch + 1) % args.print_freq == 0:
            # FT 阶段的日志输出
            if is_ft_stage:
                # mrad-ft 模式额外输出学习率
                if model_type == 'mrad-ft':
                    logger.info('epoch [{}/{}] [FT Stage], seg_loss: {:.4f}, total_loss: {:.4f}, lr: {:.6f}'.format(
                        epoch + 1, total_epochs, np.mean(seg_loss_list), np.mean(loss_list),
                        optimizer_ft.param_groups[0]['lr']))
                else:
                    logger.info('epoch [{}/{}] [FT Stage], seg_loss: {:.4f}, total_loss: {:.4f}'.format(
                        epoch + 1, total_epochs, np.mean(seg_loss_list), np.mean(loss_list)))
            else:
                # CLIP 阶段的日志输出
                logger.info('epoch [{}/{}] [CLIP Stage], clip_loss: {:.4f}'.format(
                    epoch + 1, total_epochs, np.mean(clip_loss_list)))

        # Save model
        # 按指定频率保存模型检查点
        if (epoch + 1) % args.save_freq == 0:
            # mrad-ft 模式：仅保存 patch_proj 和 image_proj
            if model_type == 'mrad-ft':
                ckp_path = os.path.join(args.save_path, f'mrad_ft_epoch_{epoch + 1}.pth')
                torch.save({
                    "image_proj": image_proj.state_dict(),
                    "patch_proj": patch_proj.state_dict()
                }, ckp_path)
            else:  # mrad-clip
                # mrad-clip 模式：保存所有可训练模块
                ckp_path = os.path.join(args.save_path, f'mrad_clip_epoch_{epoch + 1}.pth')
                torch.save({
                    "prompt_learner": prompt_learner.state_dict(),
                    "prompt_proj": prompt_proj.state_dict(),
                    "image_proj": image_proj.state_dict(),
                    "patch_proj": patch_proj.state_dict()
                }, ckp_path)

    # Final save
    # 训练结束后保存最终模型
    if model_type == 'mrad-ft':
        # 保存 MRAD-FT 最终模型
        final_path = os.path.join(args.save_path, 'mrad_ft_final.pth')
        torch.save({
            "image_proj": image_proj.state_dict(),
            "patch_proj": patch_proj.state_dict()
        }, final_path)
        logger.info(f'MRAD-FT model saved to {final_path}')
    else:
        # 保存 MRAD-CLIP 最终模型
        final_path = os.path.join(args.save_path, 'mrad_clip_final.pth')
        torch.save({
            "prompt_learner": prompt_learner.state_dict(),
            "prompt_proj": prompt_proj.state_dict(),
            "image_proj": image_proj.state_dict(),
            "patch_proj": patch_proj.state_dict()
        }, final_path)
        logger.info(f'MRAD-CLIP model saved to {final_path}')

    # 保存参数
    # 将本次训练的超参数写入文本文件，便于后续复现
    params_path = os.path.join(args.save_path, 'training_params.txt')
    with open(params_path, 'w') as f:
        f.write(f"Model Type: {model_type}\n")
        f.write(f"FT Epochs: {args.ft_epochs}\n")
        f.write(f"CLIP Epochs: {args.clip_epochs}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Image Size: {args.image_size}\n")
        f.write(f"Feature Map Layers: {args.feature_map_layer}\n")
        f.write(f"Features List: {args.features_list}\n")
        f.write(f"Seed: {args.seed}\n")
    logger.info(f'Training parameters saved to {params_path}')

# 程序入口
if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser("MRAD Training", add_help=True)
    # Data paths
    # 数据集与存储路径相关参数
    parser.add_argument("--data_path", type=str, default="/home/ts-cjh/Data/MRAD/data/spot-diff/data", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoints/released', help='path to save results')
    parser.add_argument("--cache_dir", type=str, default='./cache', help='directory for cache files')
    parser.add_argument("--dataset", type=str, default='visa', help="train dataset name")

    # Model parameters
    # 模型结构相关参数
    parser.add_argument("--depth", type=int, default=9, help="learnable_text_embedding_depth")
    parser.add_argument("--n_ctx", type=int, default=12, help="Prompt_length")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="learnable_text_embedding_length")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="feature map layers")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    # Training parameters
    # 训练超参数
    parser.add_argument("--model_type", type=str, default='mrad-clip', choices=['mrad-ft', 'mrad-clip'],
                        help='Model type to train: mrad-ft or mrad-clip')
    parser.add_argument("--ft_epochs", type=int, default=1, help="epochs for FT stage (patch_proj training)")
    parser.add_argument("--clip_epochs", type=int, default=5, help="epochs for CLIP stage (prompt_learner + prompt_proj training)")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--device", type=str, default='cuda:1')

    # 解析命令行参数
    args = parser.parse_args()
    # 固定随机种子以保证实验可复现
    setup_seed(args.seed)
    # 创建模型保存目录（如不存在则创建）
    os.makedirs(args.save_path, exist_ok=True)
    # 启动训练
    train(args)
