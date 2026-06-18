import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os

# ===================== generate_class_info: 根据数据集名生成类别映射表 =====================
# 输入数据集名称，返回该数据集包含的对象类别列表及 类别名->索引 的映射字典
def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    # 根据数据集名匹配对应的类别列表
    if dataset_name == 'mvtec':
        # MVTec AD 工业异常检测数据集（15个类别）
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'visa':
        # VisA 工业异常检测数据集（12个类别）
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'MPDD':
        # MPDD 工业异常检测数据集（6个类别）
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'BTAD':
        # BTAD 工业异常检测数据集（3个类别）
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM':
        # DAGM 工业纹理缺陷检测数据集（10个类别）
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name in ('SDD', 'KSDD', 'KSDD2'):
        # 钢表面缺陷检测数据集
        obj_list = ['electrical commutators']
    elif dataset_name == 'DTD':
        # 纹理描述数据集（12个纹理类别）
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name == 'colon':
        # 结肠相关数据集
        obj_list = ['colon']
    elif dataset_name == 'ISBI':
        # ISBI 皮肤数据集
        obj_list = ['skin']
    elif dataset_name == 'Chest':
        # 胸部影像数据集
        obj_list = ['chest']
    elif dataset_name == 'thyroid':
        # 甲状腺影像数据集
        obj_list = ['thyroid']
    elif dataset_name == 'HeadCT':
        # 头部 CT 数据集
        obj_list = ['brain']
    elif dataset_name == 'BrainMRI':
        # 脑部 MRI 数据集
        obj_list = ['brain']
    elif dataset_name == 'Br35':
        # Br35 脑部数据集
        obj_list = ['brain']
    elif dataset_name in ('ClinicDB', 'ColonDB', 'Kvasir', 'Endo','CVC300'):
        # 多种结肠镜图像数据集，统一使用 colon 标签
        obj_list = ['colon']
    elif dataset_name == 'isic':
        # ISIC 皮肤镜图像数据集
        obj_list = ['skin']
    elif dataset_name == 'tn3k':
        # TN3K 甲状腺超声数据集
        obj_list = ['thyroid']
    elif dataset_name == 'covid':
        # COVID-19 胸部影像数据集
        obj_list = ['chest']
    elif dataset_name == 'liver_ct':
        # 肝脏 CT 数据集
        obj_list = ['liver_ct']
    # 遍历类别列表，建立 类别名 -> 索引 映射
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

# ===================== Dataset: 自定义数据集类，加载图像和掩码 =====================
class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        # 读取 meta.json 获取数据集元信息（图像路径、掩码路径、类别等）
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        # 收集所有类别的数据条目到 data_all 列表中
        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        # 生成类别列表和类别映射
        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)

    def __len__(self):
        # 返回数据集总样本数
        return self.length

    def __getitem__(self, index):
        # 获取第 index 个样本的元信息
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        # 加载原始图像
        img = Image.open(os.path.join(self.root, img_path))
        # 根据是否为异常样本生成掩码
        if anomaly == 0:
            # 正常样本：生成全零掩码（无异常区域）
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            # 异常样本：尝试加载真实掩码
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # 掩码路径为目录时（仅分类任务，不报错），生成全零掩码
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                # 加载掩码文件，二值化处理（>0 设为 255）
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        # transforms: 对图像和掩码分别应用预处理和目标变换
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(   
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        # 返回字典，包含图像、掩码、类别名、异常标志、图像路径、类别ID
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}    
