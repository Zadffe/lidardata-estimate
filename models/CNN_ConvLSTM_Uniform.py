import torch
import torch.nn as nn

from models.CNN_ConvbLSTM import ConvLSTM, FrameEncoder


class CartesianCoordinateChannelsMixin:
    def _init_coord_cache(self):
        self._coord_cache_key = None
        self.register_buffer("_xx_cache", torch.empty(0), persistent=False)
        self.register_buffer("_yy_cache", torch.empty(0), persistent=False)

    def _build_coord_channels(self, batch_size, time_steps, height, width, device, dtype):
        cache_key = (height, width, str(device), str(dtype))
        if self._coord_cache_key != cache_key or self._xx_cache.numel() == 0:
            y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
            x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")

            self._xx_cache = xx.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self._yy_cache = yy.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self._coord_cache_key = cache_key

        xx = self._xx_cache.expand(batch_size, 1, time_steps, height, width)
        yy = self._yy_cache.expand(batch_size, 1, time_steps, height, width)
        return xx, yy

    def _augment_input_channels(self, x):
        batch_size, _, time_steps, height, width = x.shape
        xx, yy = self._build_coord_channels(
            batch_size=batch_size,
            time_steps=time_steps,
            height=height,
            width=width,
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat([x, xx, yy], dim=1)


class UniformGridConvLSTMNet(CartesianCoordinateChannelsMixin, nn.Module):
    def __init__(self, convlstm_hidden=64, convlstm_layers=1, temporal_pool="mean", temporal_stride=2):
        super().__init__()

        self.frame_encoder = FrameEncoder(in_channels=3, base_channels=32)
        self.temporal_model = ConvLSTM(
            input_dim=self.frame_encoder.out_channels,
            hidden_dim=convlstm_hidden,
            kernel_size=3,
            num_layers=convlstm_layers,
        )

        self.temporal_pool = temporal_pool.lower()
        if self.temporal_pool not in ("mean", "max"):
            raise ValueError("temporal_pool must be 'mean' or 'max'")

        self.temporal_stride = max(1, int(temporal_stride))
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._init_coord_cache()

        self.head_dir = nn.Sequential(
            nn.Linear(convlstm_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )
        self.head_hs = nn.Sequential(
            nn.Linear(convlstm_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, 1, T, H, W]
        bsz, _, time_steps, height, width = x.shape

        if self.temporal_stride > 1:
            x = x[:, :, :: self.temporal_stride, :, :]
            time_steps = x.size(2)

        x = self._augment_input_channels(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bsz * time_steps, 3, height, width)
        frame_feat = self.frame_encoder(x)
        _, c_feat, h_feat, w_feat = frame_feat.shape

        frame_feat = frame_feat.view(bsz, time_steps, c_feat, h_feat, w_feat)
        seq_feat = self.temporal_model(frame_feat)

        if self.temporal_pool == "mean":
            temporal_feat = seq_feat.mean(dim=1)
        else:
            temporal_feat = seq_feat.max(dim=1).values

        global_feat = self.spatial_pool(temporal_feat).flatten(1)
        pred_dir = self.head_dir(global_feat)
        pred_hs = self.head_hs(global_feat)
        pred_dir = torch.nn.functional.normalize(pred_dir, p=2, dim=1)
        return pred_dir, pred_hs


def build_uniform_convlstm_model(cfg):
    return UniformGridConvLSTMNet(
        convlstm_hidden=getattr(cfg, "convlstm_hidden", 64),
        convlstm_layers=getattr(cfg, "convlstm_layers", 1),
        temporal_pool=getattr(cfg, "temporal_pool", "mean"),
        temporal_stride=getattr(cfg, "temporal_stride", 2),
    )


def build_model(cfg):
    return build_uniform_convlstm_model(cfg)
