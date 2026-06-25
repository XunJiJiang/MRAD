import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# GCNLayer: 图卷积层，基于度归一化的均值聚合
# 对每个节点的邻居特征做均值聚合后经线性变换更新节点表示
# ============================================================
class GCNLayer(nn.Module):
    # 初始化：输入维度、输出维度、dropout 比率
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        # 线性变换层：映射邻居聚合后的特征
        self.linear = nn.Linear(in_dim, out_dim)
        # Dropout 层：正则化，防止过拟合
        self.dropout = nn.Dropout(dropout)

    # 前向传播：edge_index 为 [2, E] 格式，x 为 [N, in_dim] 节点特征
    def forward(self, x, edge_index):
        # 获取源节点索引和目标节点索引
        src, dst = edge_index[0], edge_index[1]
        # 将源节点的特征按目标节点分组，初始化聚合结果全零
        out = torch.zeros_like(x)
        # scatter_add：将 src 特征累加到对应的 dst 位置
        out = out.scatter_add(0, dst.unsqueeze(-1).expand(-1, x.size(1)), x[src])
        # 计算每个目标节点的入度（邻居数量）
        deg = torch.zeros(x.size(0), device=x.device).scatter_add(0, dst, torch.ones(src.size(0), device=x.device))
        # clamp 防止除零，度归一化：除以入度
        deg = deg.clamp(min=1)
        out = out / deg.unsqueeze(-1)
        # 线性变换 + ReLU 激活 + Dropout
        out = self.linear(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out


# ============================================================
# GATLayer: 图注意力层，多头点积注意力聚合
# 每个注意力头独立计算源-目标节点间的注意力权重，加权聚合邻居特征
# ============================================================
class GATLayer(nn.Module):
    # 初始化：输入维度、输出维度（每个头的输出维度）、头数、dropout 比率、负斜率
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, negative_slope=0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.negative_slope = negative_slope

        # 每个头独立的查询（Q）和键（K）投影矩阵
        self.W_q = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim * heads, bias=False)
        # Dropout 层：应用于注意力权重和输出
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)
        # 可学习的缩放因子
        self.scale = out_dim ** -0.5

    # 前向传播：x 为 [N, in_dim] 节点特征，edge_index 为 [2, E]
    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)

        # 计算所有节点的 Q 和 K，reshape 为 [N, heads, out_dim]
        Q = self.W_q(x).view(N, self.heads, self.out_dim)
        K = self.W_k(x).view(N, self.heads, self.out_dim)

        # 取出源节点和目标节点对应的 Q、K
        Q_dst = Q[dst]  # [E, heads, out_dim]
        K_src = K[src]  # [E, heads, out_dim]

        # 计算点积注意力分数：[E, heads]，除以 sqrt(d_k) 缩放
        attn_scores = (Q_dst * K_src).sum(dim=-1) * self.scale

        # LeakyReLU 激活，负斜率为 negative_slope
        attn_scores = F.leaky_relu(attn_scores, negative_slope=self.negative_slope)

        # 对每个目标节点做 softmax 归一化（使用 scatter 实现分段 softmax）
        # 计算每个目标节点的分数最大值，用于数值稳定性
        score_max = torch.zeros(N, self.heads, device=x.device)
        score_max = score_max.scatter_reduce(0, dst.unsqueeze(-1).expand(-1, self.heads), attn_scores, reduce='amax', include_self=False)
        # exp(score - max)，数值稳定
        attn_exp = torch.exp(attn_scores - score_max[dst])
        # 计算每个目标节点的 exp 之和（分母）
        exp_sum = torch.zeros(N, self.heads, device=x.device)
        exp_sum = exp_sum.scatter_add(0, dst.unsqueeze(-1).expand(-1, self.heads), attn_exp)
        exp_sum = exp_sum.clamp(min=1e-8)
        # 归一化注意力权重
        attn_weights = attn_exp / exp_sum[dst]
        # 对注意力权重做 dropout
        attn_weights = self.dropout_attn(attn_weights)

        # 加权聚合源节点特征：[E, heads, out_dim] 按注意力权重加权
        V_src = K_src.view(-1, self.heads, self.out_dim)  # 复用 K 作为 Value
        weighted = V_src * attn_weights.unsqueeze(-1)       # [E, heads, out_dim]

        # scatter_add 聚合同一目标节点的所有邻居
        out = torch.zeros(N, self.heads, self.out_dim, device=x.device)
        idx = dst.unsqueeze(-1).unsqueeze(-1).expand(-1, self.heads, self.out_dim)
        out = out.scatter_add(0, idx, weighted)

        # 多头拼接：[N, heads * out_dim]
        out = out.view(N, self.heads * self.out_dim)
        out = self.dropout_out(out)
        return out


# ============================================================
# GraphMemoryBank: 图神经网络增强记忆库
# 对记忆原型构建 k-NN 图，通过 GCN/GAT 层进行消息传递更新原型表示
# 采用残差连接保留原始原型信息，最终 L2 归一化
# ============================================================
class GraphMemoryBank(nn.Module):
    # 初始化参数：
    #   in_dim: 原型特征维度（ViT patch features 为 1024）
    #   hidden_dim: GNN 隐层维度，默认 512
    #   out_dim: 输出维度，默认与 in_dim 相同（1024）
    #   num_layers: GNN 层数，默认 2
    #   gnn_type: GNN 类型，'gcn' 或 'gat'，默认 'gat'
    #   heads: GAT 注意力头数（仅 gnn_type='gat' 时生效），默认 4
    #   topk_neighbors: k-NN 图中每个节点的邻居数，默认 10
    #   dropout: Dropout 比率，默认 0.1
    def __init__(self, in_dim=1024, hidden_dim=512, out_dim=1024,
                 num_layers=2, gnn_type='gat', heads=4,
                 topk_neighbors=10, dropout=0.1):
        super().__init__()
        self.topk_neighbors = topk_neighbors
        self.num_layers = num_layers
        self.gnn_type = gnn_type

        # 输入投影层：将 in_dim 映射到 hidden_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # 构建 GNN 层列表
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            # GAT 模式：每个头的输出维度 = hidden_dim // heads
            if gnn_type == 'gat':
                self.gnn_layers.append(
                    GATLayer(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
                )
            # GCN 模式：输入输出维度均为 hidden_dim
            elif gnn_type == 'gcn':
                self.gnn_layers.append(
                    GCNLayer(hidden_dim, hidden_dim, dropout=dropout)
                )

        # 输出投影层：将 hidden_dim 映射回 out_dim（= in_dim），用于残差连接
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    # _build_edge_index: 基于余弦相似度构建 k-NN 图的边索引
    # 输入：x 为 [N, D] 节点特征（L2 归一化后）
    # 输出：edge_index [2, E]，每列 (src, dst)，src→dst
    def _build_edge_index(self, x):
        N = x.size(0)
        # k 不能超过节点总数-1（排除自身）
        k = min(self.topk_neighbors, N - 1)
        # 计算余弦相似度矩阵：[N, N]
        sim = torch.matmul(x, x.t())
        # 将对角线（自身相似度）置为 -inf，排除自身
        sim.fill_diagonal_(float('-inf'))
        # 取 top-k 相似邻居：返回分数和索引
        _, topk_indices = torch.topk(sim, k, dim=-1)  # [N, k]
        # 构建 edge_index：[2, N*k]
        dst = torch.arange(N, device=x.device).unsqueeze(1).expand(N, k).reshape(-1)
        src = topk_indices.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)
        return edge_index

    # 前向传播：
    #   输入：prototypes [K, in_dim] 记忆原型
    #   输出：refined [K, out_dim] 图增强后的记忆原型
    def forward(self, prototypes):
        # 保存原始输入用于最终残差连接
        original = prototypes  # [K, in_dim]
        # L2 归一化后构建 k-NN 图
        x_norm = F.normalize(prototypes, p=2, dim=-1)
        # 构建边索引
        edge_index = self._build_edge_index(x_norm)

        # 输入投影：in_dim → hidden_dim
        x = self.input_proj(prototypes)

        # 逐层 GNN 消息传递，每层后接残差连接
        for i, layer in enumerate(self.gnn_layers):
            h = layer(x, edge_index)
            # GCN 输出维度为 hidden_dim，可直接残差
            if self.gnn_type == 'gcn':
                x = x + h
            # GAT 多头拼接后维度为 hidden_dim（heads * (hidden_dim // heads) = hidden_dim），可直接残差
            elif self.gnn_type == 'gat':
                x = x + h
            # 非最后一层时做 L2 归一化，避免特征尺度爆炸
            if i != self.num_layers - 1:
                x = F.normalize(x, p=2, dim=-1)

        # 输出投影：hidden_dim → out_dim，并加原始原型残差
        refined = self.output_proj(x) + original
        # 最终 L2 归一化，确保与后续检索的余弦相似度计算兼容
        refined = F.normalize(refined, p=2, dim=-1)
        return refined
