import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from open_clip import tokenizer
# simple_tokenizer = tokenizer.SimpleTokenizer()
from copy import deepcopy
import torch.nn as nn
from collections import OrderedDict
# 初始化全局分词器实例
_tokenizer = _Tokenizer()

# LayerNorm: 继承torch的LayerNorm，处理fp16精度问题
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    # 前向传播：将输入转为float32再调用父类LayerNorm，最后转回原始类型
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


# QuickGELU: GELU激活函数的近似实现，使用sigmoid乘以输入
class QuickGELU(nn.Module):
    # 前向传播：x * sigmoid(1.702 * x)，近似标准GELU
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# tokenize: 将文本字符串转换为token序列（CLIP分词）
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    # 如果输入是单个字符串，转为列表以便统一处理
    if isinstance(texts, str):
        texts = [texts]

    # 获取起始符和结束符的token ID
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    # 为每个文本生成完整token序列：[SOT] + 编码 + [EOT]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    # 根据torch版本选择LongTensor或IntTensor
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    # 遍历每个文本的token序列，填充到结果张量中
    for i, tokens in enumerate(all_tokens):
        # 如果token长度超过上下文长度
        if len(tokens) > context_length:
            # 允许截断：保留前context_length个token，末尾强制设为EOT
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            # 不允许截断：抛出异常
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        # 将token填入结果张量的对应行
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

# encode_text_with_prompt_ensemble: 使用提示集合对文本进行CLIP编码，返回正常/异常两类文本特征
def encode_text_with_prompt_ensemble(model, texts, device):
    # 正常状态提示模板列表
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    # 异常状态提示模板列表
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    # 将正常和异常提示模板组合为状态列表
    prompt_state = [prompt_normal, prompt_abnormal]
    # 图像描述模板（CoOp风格），用于组合状态描述生成完整句子
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_features = []
    # 遍历正常和异常两种状态
    for i in range(len(prompt_state)):
        # 将类别名填入当前状态的提示模板中
        prompted_state = [state.format(texts[0]) for state in prompt_state[i]]
        prompted_sentence = []
        # 遍历每个状态提示，与所有图像模板组合生成完整句子
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        # 将生成的句子列表进行分词编码
        prompted_sentence = tokenize(prompted_sentence)
        # 通过CLIP文本编码器获取文本嵌入
        class_embeddings = model.encode_text(prompted_sentence.to(device))
        # 对嵌入进行L2归一化
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        # 取所有模板嵌入的均值作为该状态的文本特征
        class_embedding = class_embeddings.mean(dim=0)
        # 再次归一化
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)

    # 堆叠正常和异常特征，转置得到 [2, D] 形状
    text_features = torch.stack(text_features, dim=1).to(device).t()

    return text_features



# _get_clones: 将指定模块深度复制N次，返回ModuleList
def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

# AnomalyCLIP_PromptLearner: 异常检测CLIP的可学习提示学习器
# 核心功能：维护正类（正常）和负类（异常）两组可学习的上下文向量（prompt），
# 并通过MLP和投影层将偏置信息融合到提示中，最终生成用于异常检测的文本提示
class AnomalyCLIP_PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()
        # 类别名列表（固定为"object"）
        classnames = ["object"]
        self.n_cls = len(classnames)
        # 可学习提示token的长度
        self.n_ctx = design_details["Prompt_length"]
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        # 深层文本嵌入的可学习长度
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"] 
        # 初始化上下文文本（空字符串表示随机初始化）
        ctx_init_pos = ""
        ctx_init_neg = ""
        # 获取CLIP模型的数据类型
        dtype = clip_model.transformer.get_cast_dtype()

        # 上下文向量的维度（与CLIP最终LayerNorm权重维度一致）
        ctx_dim = clip_model.ln_final.weight.shape[0]

        
        self.classnames = classnames

        # 正常状态的提示模板列表
        self.state_normal_list = [
            "{}",
        ]

        # 异常状态的提示模板列表
        self.state_anomaly_list = [
            "damaged {}",
        ]
        
        # 正常和异常状态模板的数量
        normal_num = len(self.state_normal_list)
        anormaly_num = len(self.state_anomaly_list)
        self.normal_num = normal_num
        self.anormaly_num = anormaly_num

        # 如果提供了初始化的上下文文本（非空），则使用给定文本初始化上下文向量
        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            # 将下划线替换为空格
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            # 根据空格分割计算实际token长度
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            #初始化text成bpd编码
            prompt_pos = tokenize(ctx_init_pos)
            prompt_neg = tokenize(ctx_init_neg)
            # 在不计算梯度的上下文中生成对应的文本嵌入
            with torch.no_grad():
                #生成相应的text embedding
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            #这些是去除出来EOS 和 # CLS, EOS， 获得可学习的textual prompt
            # 提取除SOS和EOS之外的中间token嵌入作为可学习上下文向量
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            # 为每个类别复制上下文向量
            if True:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(self.n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        # 否则进行随机初始化
        else:
            # Random Initialization
            # 类别特定初始化：为每个类别创建独立的上下文向量
            if True:
                print("Initializing class-specific contexts")
                #这里是cls是类的个数，n_ctx_pos代表learnable token的长度，ctx_dim表示prompt的dimension
                ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(self.n_cls, self.anormaly_num, n_ctx_neg, ctx_dim, dtype=dtype)
            # 通用上下文初始化（不使用）
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            # 用正态分布初始化上下文向量
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            # 用"X"占位符构造提示前缀
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)

        # 复合提示的深度（层数）
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        # 为每一层复合提示创建可学习参数（除最后一层外）
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        # 正态初始化每层复合提示参数
        for single_para in self.compound_prompts_text:
            print("single_para", single_para.shape)
            nn.init.normal_(single_para, std=0.02)

        # 为每层复合提示创建对应的线性投影层（将ctx_dim投影到896维）
        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)


        # 正类（正常）上下文向量，将被优化
        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        # 负类（异常）上下文向量，将被优化
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        # 将类别名中的下划线替换为空格
        classnames = [name.replace("_", " ") for name in classnames]
        # 计算每个类别名的token数量
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        d_model = 768
        # MLP模块：用于处理文本特征（线性扩展 -> GELU -> 线性压缩）
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))

        # 生成正类提示文本：前缀 + 状态模板填充类别名
        prompts_pos = [prompt_prefix_pos +  " " + template.format(name)+ "." for template in self.state_normal_list for name in classnames]
        # 生成负类提示文本：前缀 + 状态模板填充类别名
        prompts_neg = [prompt_prefix_neg +  " " + template.format(name)+ "." for template in self.state_anomaly_list for name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
     
        # 对每个正类提示文本进行分词
        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        # 对每个负类提示文本进行分词
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))
        # 拼接所有分词结果
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        #生成相应的text embedding
        # 不计算梯度，生成提示的前缀和后缀嵌入作为固定缓冲区
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            print("embedding_pos", embedding_pos.shape)
            # 重排嵌入形状：[模板数, 类别数, 序列长度, 维度] -> [类别数, 模板数, 序列长度, 维度]
            embedding_pos = embedding_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)


        # 注册固定缓冲区：正类提示的前缀嵌入（SOS token）
        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :] )
        # 注册固定缓冲区：正类提示的后缀嵌入（除SOS和可学习上下文之外的部分）
        self.register_buffer("token_suffix_pos", embedding_pos[:, :,1 + n_ctx_pos:, :])
        # 注册固定缓冲区：负类提示的前缀嵌入（SOS token）
        self.register_buffer("token_prefix_neg", embedding_neg[:,:, :1, :])
        # 注册固定缓冲区：负类提示的后缀嵌入
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :])

        # 重排正类分词结果：[模板数, 类别数, 序列长度] -> [类别数, 模板数, 序列长度]
        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)

        # 重排负类分词结果
        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(anormaly_num, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        # tokenized_prompts = torch.cat([tokenized_prompts_pos, tokenized_prompts_neg], dim=0)  # torch.Tensor
        # 注册分词后的正类提示作为固定缓冲区
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        # 注册分词后的负类提示作为固定缓冲区
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)



    # 前向传播：根据偏置信息生成正类和负类的提示嵌入
    def forward(self, cls_id =None,bias=None):
        
        batch, n, d_model = bias.shape
        # 提取正类偏置，扩展维度便于广播加法
        bia_pos = bias[:,0,:].unsqueeze(1).unsqueeze(2)    
        # 提取负类偏置，扩展维度便于广播加法
        bia_neg = bias[:,1,:].unsqueeze(1).unsqueeze(2)

        # 正类上下文向量加上偏置，融合异常检测的patch信息
        ctx_pos = self.ctx_pos + bia_pos
        # ctx_pos = self.ctx_pos
        # ctx_pos = ctx_pos.expand(bia_pos.shape[0], -1, -1, -1).clone()  # [B, 1, 12, 768]
        # # 分离前6和后6
        # ctx_pos_front = ctx_pos[:, :, :6, :]          # 不加偏置的前6
        # ctx_pos_back = ctx_pos[:, :, 6:, :] + bia_pos # 后6加偏置
        # # 拼接回来
        # ctx_pos = torch.cat([ctx_pos_front, ctx_pos_back], dim=2)  # [B, 1, 12, 768]

        # 负类上下文向量加上偏置
        ctx_neg = self.ctx_neg + bia_neg
        # ctx_neg = self.ctx_neg
        # ctx_neg = ctx_neg.expand(bia_neg.shape[0], -1, -1, -1).clone()  # [B, 1, 12, 768]       
        # ctx_neg_front = ctx_neg[:, :, :6, :]          # 不加偏置的前6
        # ctx_neg_back = ctx_neg[:, :, 6:, :] + bia_neg # 后6加偏置
        # ctx_neg = torch.cat([ctx_neg_front, ctx_neg_back], dim=2)  # [B, 1, 12, 768]

        # print("shape", self.ctx_pos[0:1].shape, ctx_pos.shape)
        # 将固定前缀和后缀缓冲区扩展到当前batch大小
        prefix_pos = self.token_prefix_pos.expand(batch,1,1,768)
        prefix_neg = self.token_prefix_neg.expand(batch,1,1,768)
        suffix_pos = self.token_suffix_pos.expand(batch,1,64,768)
        suffix_neg = self.token_suffix_neg.expand(batch,1,64,768)

        # print(prefix_pos.shape, prefix_neg.shape)

        # 拼接正类提示的完整token序列：前缀 + 可学习上下文 + 后缀
        prompts_pos = torch.cat(
            [
                # N(the number of template), 1, dim
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=2,
        )

        # 拼接负类提示的完整token序列：前缀 + 可学习上下文 + 后缀
        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=2,
        )
        # 压平正类提示的形状：[B, 模板数, 序列长度, 维度] -> [B*模板数, 序列长度, 维度]
        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)
        # 压平负类提示的形状
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)
        # 将正类和负类提示堆叠，形状为 [B*模板数, 2, 序列长度, 维度]
        prompts = torch.stack([prompts_pos, prompts_neg], dim=1)
        

        # 压平分词后的正类提示
        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1,  d)
        # 压平分词后的负类提示
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1,  d)
        # 拼接正负类分词结果
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim = 0)


        # 返回：提示嵌入、分词结果、复合提示文本参数
        return prompts, tokenized_prompts, self.compound_prompts_text
