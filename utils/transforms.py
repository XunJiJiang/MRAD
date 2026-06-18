import torchvision.transforms as transforms
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from AnomalyCLIP_lib.transform import image_transform
from AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


# ===================== normalize: 最小-最大归一化 =====================
# 将预测结果归一化到 [0, 1] 区间
def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        # 未提供最大/最小值时，使用数据自身的极值进行归一化
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        # 提供了最大/最小值时，按给定的范围进行归一化
        return (pred - min_value) / (max_value - min_value)

# ===================== get_transform: 获取图像预处理和目标变换 =====================
# 根据配置参数构造图像预处理 pipeline 和掩码目标变换 pipeline
def get_transform(args):
    # 使用 CLIP 的图像预处理（含归一化均值和标准差）
    preprocess = image_transform(args.image_size, is_train=False, mean = OPENAI_DATASET_MEAN, std = OPENAI_DATASET_STD)
    # 目标（掩码）变换：缩放 -> 中心裁剪 -> 转 Tensor
    target_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])
    # 替换预处理 pipeline 中的 Resize 和 CenterCrop 以匹配目标尺寸
    preprocess.transforms[0] = transforms.Resize(size=(args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                                    max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(args.image_size, args.image_size))
    return preprocess, target_transform
