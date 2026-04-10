import torch
import torch.nn as nn

from models.vision_transformer import build_temporal_transformer_model


class FrameEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(FrameEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.out_channels = base_channels * 4

    def forward(self, x):
        return self.features(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x_t, h_cur, c_cur):
        combined = torch.cat([x_t, h_cur], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device, dtype):
        height, width = spatial_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList()

        for layer_index in range(num_layers):
            cur_input_dim = input_dim if layer_index == 0 else hidden_dim
            self.cells.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size=kernel_size))

    def forward(self, x):
        # x: [B, T, C, H, W]
        batch_size, time_steps, _, height, width = x.shape
        device = x.device
        dtype = x.dtype

        layer_input = x
        for cell in self.cells:
            h, c = cell.init_hidden(batch_size, (height, width), device, dtype)
            outputs = []
            for t in range(time_steps):
                h, c = cell(layer_input[:, t], h, c)
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=1)

        return layer_input


class CoordinateChannelsMixin:
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


class MultiTaskWaveNet(CoordinateChannelsMixin, nn.Module):
    def __init__(self, convlstm_hidden=64, convlstm_layers=1, temporal_pool="mean", temporal_stride=2):
        super(MultiTaskWaveNet, self).__init__()

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


class MultiTaskWaveNetPureCNN(CoordinateChannelsMixin, nn.Module):
    def __init__(self, temporal_pool="mean", temporal_stride=2):
        super(MultiTaskWaveNetPureCNN, self).__init__()

        self.frame_encoder = FrameEncoder(in_channels=3, base_channels=32)

        self.temporal_pool = temporal_pool.lower()
        if self.temporal_pool not in ("mean", "max"):
            raise ValueError("temporal_pool must be 'mean' or 'max'")

        self.temporal_stride = max(1, int(temporal_stride))
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._init_coord_cache()

        feat_dim = self.frame_encoder.out_channels
        self.head_dir = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )
        self.head_hs = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        bsz, _, time_steps, height, width = x.shape

        if self.temporal_stride > 1:
            x = x[:, :, :: self.temporal_stride, :, :]
            time_steps = x.size(2)

        x = self._augment_input_channels(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bsz * time_steps, 3, height, width)
        frame_feat = self.frame_encoder(x)
        _, c_feat, h_feat, w_feat = frame_feat.shape

        frame_feat = frame_feat.view(bsz, time_steps, c_feat, h_feat, w_feat)

        if self.temporal_pool == "mean":
            temporal_feat = frame_feat.mean(dim=1)
        else:
            temporal_feat = frame_feat.max(dim=1).values

        global_feat = self.spatial_pool(temporal_feat).flatten(1)
        pred_dir = self.head_dir(global_feat)
        pred_hs = self.head_hs(global_feat)
        pred_dir = torch.nn.functional.normalize(pred_dir, p=2, dim=1)
        return pred_dir, pred_hs


def build_model(cfg):
    model_name = str(getattr(cfg, "model_name", "ConvLSTM")).lower()

    if model_name in ("convlstm", "cnn_convlstm", "cnn+convlstm"):
        return MultiTaskWaveNet(
            convlstm_hidden=getattr(cfg, "convlstm_hidden", 64),
            convlstm_layers=getattr(cfg, "convlstm_layers", 1),
            temporal_pool=getattr(cfg, "temporal_pool", "mean"),
            temporal_stride=getattr(cfg, "temporal_stride", 2),
        )

    if model_name in ("purecnn", "cnn", "cnn_only", "pure_cnn"):
        return MultiTaskWaveNetPureCNN(
            temporal_pool=getattr(cfg, "temporal_pool", "mean"),
            temporal_stride=getattr(cfg, "temporal_stride", 2),
        )

    if model_name in (
        "temporaltransformer",
        "cnn_temporal_transformer",
        "cnn+transformer",
        "transformer",
        "vit",
    ):
        return build_temporal_transformer_model(cfg)

    raise ValueError(
        f"Unsupported model_name={getattr(cfg, 'model_name', None)}. "
        f"Use 'ConvLSTM', 'PureCNN', or 'TemporalTransformer'."
    )
