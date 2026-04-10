import math

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
CNN Stem + Temporal Transformer
用于 LiDAR 波高 / 波向联合估计的混合模型。

整体流程
1. 输入张量形状为 [B, 1, T, H, W]
   - B: batch size
   - 1: 原始输入通道，只包含高度场 z
   - T: 时间帧数
   - H, W: 空间网格尺寸
2. 在模型内部补充两个确定性的几何先验通道
   - rr: 到图像中心的归一化径向距离
   - theta: 极角
   于是单帧输入从 1 通道变成 3 通道 [z, rr, theta]
3. 逐帧经过 CNN stem
   - 提取局部空间纹理、波面结构和邻域模式
   - 同时完成一定的空间下采样，降低后续时序建模的成本
4. 对每一帧的特征图做空间池化，压缩为一个 token
   - 得到形状 [B, T, C] 的时间序列特征
5. 送入 Temporal Transformer
   - 在时间维上建模长程依赖
   - 学习不同帧之间的传播、相位变化和整体动态关系
6. 通过两个任务头输出结果
   - 波向头：输出 [sin(dir), cos(dir)]
   - 波高头：输出归一化后的 Hs，范围约束在 [0, 1]

为什么采用这种混合结构
1. 纯 Transformer 直接处理原始网格，通常对数据量要求更高。
2. CNN 对局部空间结构有更强的归纳偏置，更适合当前这种规则栅格输入。
3. Transformer 在时序建模上更灵活，能替代 ConvLSTM 去捕获长距离时间依赖。
4. 保留 rr / theta 先验，有助于维持当前项目里已经验证有效的几何信息表达。
"""


class ConvBNGELU(nn.Module):
    """
    最基本的卷积特征块。

    结构:
    Conv2d -> BatchNorm2d -> GELU

    作用:
    - 用卷积提取局部空间特征
    - 用 BN 稳定训练
    - 用 GELU 提供更平滑的非线性
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class CNNStem(nn.Module):
    """
    Transformer 前面的逐帧空间编码器。

    输入输出:
    - 输入: [B * T, C, H, W]
    - 输出: [B * T, embed_dim, H', W']

    设计目的:
    1. 在单帧内提取局部空间模式，而不是一开始就让 Transformer 处理原始像素网格
    2. 通过池化逐步缩小空间分辨率，减少后续模型的计算压力
    3. 保留足够的语义信息，使每一帧最终可以压缩成一个高质量 token
    """

    def __init__(self, in_channels=3, base_channels=32, embed_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNGELU(in_channels, base_channels),
            ConvBNGELU(base_channels, base_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNGELU(base_channels, base_channels * 2),
            ConvBNGELU(base_channels * 2, base_channels * 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNGELU(base_channels * 2, embed_dim),
            ConvBNGELU(embed_dim, embed_dim),
        )
        self.out_channels = embed_dim

    def forward(self, x):
        return self.features(x)


class CoordinateChannelsMixin:
    """
    为每一帧补充两个确定性的几何先验通道。

    通道定义:
    - rr: 像素到图像中心的归一化径向距离
    - theta: 像素对应的归一化极角

    这样做的目的:
    1. 给模型显式提供“位置在哪里”的信息
    2. 给模型显式提供“相对于中心的方向”信息
    3. 与现有 CNN + ConvLSTM 模型保持一致，方便做公平对比
    """

    def _init_coord_cache(self):
        self._coord_cache_key = None
        self.register_buffer("_rr_cache", torch.empty(0), persistent=False)
        self.register_buffer("_theta_cache", torch.empty(0), persistent=False)

    def _build_coord_channels(self, batch_size, time_steps, height, width, device, dtype):
        cache_key = (height, width, str(device), str(dtype))
        if self._coord_cache_key != cache_key or self._rr_cache.numel() == 0:
            y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
            x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")

            rr = torch.sqrt(xx * xx + yy * yy).clamp(0.0, 1.0)
            theta = torch.atan2(yy, xx) / torch.pi

            self._rr_cache = rr.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self._theta_cache = theta.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self._coord_cache_key = cache_key

        rr = self._rr_cache.expand(batch_size, 1, time_steps, height, width)
        theta = self._theta_cache.expand(batch_size, 1, time_steps, height, width)
        return rr, theta

    def _augment_input_channels(self, x):
        batch_size, _, time_steps, height, width = x.shape
        rr, theta = self._build_coord_channels(
            batch_size=batch_size,
            time_steps=time_steps,
            height=height,
            width=width,
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat([x, rr, theta], dim=1)


def build_sinusoidal_temporal_encoding(seq_len, dim, device, dtype):
    """
    构造时间维上的正弦位置编码。

    这里使用固定的 sinusoidal encoding，而不是可学习的位置表，原因是:
    1. 对不同长度的时间序列更稳健
    2. 参数量更小
    3. 便于后续在不同帧数设置下复用
    """

    position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(dim, 1))
    )

    pe = torch.zeros(seq_len, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)

    if dim > 1:
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

    return pe.unsqueeze(0).to(dtype=dtype)


class MultiTaskWaveCNNTemporalTransformer(CoordinateChannelsMixin, nn.Module):
    """
    混合式多任务模型:
    逐帧 CNN 编码 -> 时间 Transformer -> 多任务回归头

    输入:
    - x: [B, 1, T, H, W]

    输出:
    - pred_dir: [B, 2]
      表示 [sin(dir), cos(dir)]，最后会被归一化到单位圆上
    - pred_hs: [B, 1]
      表示归一化后的 Hs，范围在 [0, 1]
    """

    def __init__(
        self,
        embed_dim=128,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.1,
        temporal_pool="mean",
        temporal_stride=2,
    ):
        super().__init__()

        # 时间维聚合方式:
        # - mean: 对所有时间 token 求平均
        # - max:  对所有时间 token 取最大响应
        # - cls:  使用额外的 CLS token 作为全局时序表示
        self.temporal_pool = temporal_pool.lower()
        if self.temporal_pool not in ("mean", "max", "cls"):
            raise ValueError("temporal_pool must be 'mean', 'max', or 'cls'")

        self.temporal_stride = max(1, int(temporal_stride))
        self.frame_encoder = CNNStem(in_channels=3, base_channels=32, embed_dim=embed_dim)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.input_dropout = nn.Dropout(dropout)
        self._init_coord_cache()

        # Transformer 只沿时间维工作。
        # 每一帧先被压成一个 token，因此这里建模的是“帧与帧之间”的关系。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.temporal_norm = nn.LayerNorm(embed_dim)

        # 如果选择 cls 聚合，则引入一个额外的全局 token。
        if self.temporal_pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.register_parameter("cls_token", None)

        # 波向分支:
        # 输出二维向量，后续会归一化为单位向量，分别对应 sin / cos。
        self.head_dir = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(attn_dropout),
            nn.Linear(256, 2),
        )
        # 波高分支:
        # 输出单值，并通过 Sigmoid 限制在 [0, 1]，与当前数据归一化方式一致。
        self.head_hs = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(attn_dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                if getattr(module, "weight", None) is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def _pool_temporal_tokens(self, tokens):
        if self.temporal_pool == "cls":
            return tokens[:, 0]

        if self.temporal_pool == "mean":
            return tokens.mean(dim=1)

        return tokens.max(dim=1).values

    def forward(self, x):
        # 外部输入约定:
        # x 的形状为 [B, 1, T, H, W]
        batch_size, _, time_steps, height, width = x.shape

        if self.temporal_stride > 1:
            x = x[:, :, :: self.temporal_stride, :, :]
            time_steps = x.size(2)

        # 先补充 rr / theta 两个几何先验通道，保持与现有项目一致。
        x = self._augment_input_channels(x)

        # 将张量整理成逐帧编码的形式:
        # [B, 3, T, H, W] -> [B*T, 3, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * time_steps, 3, height, width)
        frame_feat = self.frame_encoder(x)

        # 每一帧的空间特征图通过全局平均池化压成一个 token。
        frame_tokens = self.spatial_pool(frame_feat).flatten(1)
        frame_tokens = frame_tokens.view(batch_size, time_steps, self.frame_encoder.out_channels)

        # 为时间序列加上位置编码，让 Transformer 感知帧顺序。
        pos_embed = build_sinusoidal_temporal_encoding(
            seq_len=time_steps,
            dim=frame_tokens.size(-1),
            device=frame_tokens.device,
            dtype=frame_tokens.dtype,
        )
        frame_tokens = self.input_dropout(frame_tokens + pos_embed)

        # 可选 CLS token:
        # 让 Transformer 学习一个专门用于汇总整段序列信息的全局表示。
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(batch_size, -1, -1).to(frame_tokens.dtype)
            tokens = torch.cat([cls_token, frame_tokens], dim=1)
        else:
            tokens = frame_tokens

        tokens = self.temporal_encoder(tokens)
        tokens = self.temporal_norm(tokens)
        global_feat = self._pool_temporal_tokens(tokens)

        # 两个任务头分别输出波向和波高。
        pred_dir = self.head_dir(global_feat)
        pred_hs = self.head_hs(global_feat)

        # 将波向输出限制在单位圆上，保持与标签表达方式一致。
        pred_dir = F.normalize(pred_dir, p=2, dim=1)
        return pred_dir, pred_hs


def build_temporal_transformer_model(cfg):
    """
    根据配置文件构建 Temporal Transformer 模型。

    可选配置项:
    - vit_embed_dim
    - vit_depth
    - vit_num_heads
    - vit_mlp_ratio
    - vit_dropout
    - vit_attn_dropout
    - temporal_pool
    - temporal_stride
    """

    return MultiTaskWaveCNNTemporalTransformer(
        embed_dim=getattr(cfg, "vit_embed_dim", 128),
        depth=getattr(cfg, "vit_depth", 4),
        num_heads=getattr(cfg, "vit_num_heads", 8),
        mlp_ratio=getattr(cfg, "vit_mlp_ratio", 4.0),
        dropout=getattr(cfg, "vit_dropout", 0.1),
        attn_dropout=getattr(cfg, "vit_attn_dropout", 0.1),
        temporal_pool=getattr(cfg, "temporal_pool", "mean"),
        temporal_stride=getattr(cfg, "temporal_stride", 2),
    )
