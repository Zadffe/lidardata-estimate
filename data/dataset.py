# -*- coding: utf-8 -*-
import os
import glob
import random

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from configs.config import Config


class LidarWaveDataset(Dataset):
    def __init__(self, mode='train', augment=False):
        """
        Args:
            mode: 'train' / 'val' / 'test'
            augment: 是否开启数据增强。通常训练集开，验证/测试集关。
        """
        self.cfg = Config()
        self.mode = mode
        self.augment = augment
        self.file_list = glob.glob(os.path.join(self.cfg.data_root, mode, '*.mat'))

        if len(self.file_list) == 0:
            print(f"警告: {mode} 集未找到 .mat 文件")

    def __len__(self):
        return len(self.file_list)

    def apply_masking(self, tensor: np.ndarray) -> np.ndarray:
        """
        输入: [T, H, W] numpy 张量
        输出: 增强后的 numpy 张量
        """
        tensor = tensor.copy()
        T, H, W = tensor.shape

        # 1) 动态随机丢点：模拟回波随机缺失
        if random.random() < 0.8:
            drop_prob = random.uniform(0.2, 0.9)
            rand_mask = np.random.choice([0, 1], size=(T, H, W), p=[drop_prob, 1 - drop_prob])
            tensor = tensor * rand_mask

        # 2) 扫描线短时丢失：模拟行/列方向的数据链路丢包
        if random.random() < 0.55:
            n_lines = random.randint(1, max(1, min(H, W) // 12))
            use_rows = (random.random() < 0.5)
            for _ in range(n_lines):
                line_idx = random.randint(0, (H - 1) if use_rows else (W - 1))
                start_t = random.randint(0, T - 1)
                dur = random.randint(max(1, T // 8), max(2, T // 3))
                end_t = min(T, start_t + dur)
                if use_rows:
                    tensor[start_t:end_t, line_idx, :] = 0
                else:
                    tensor[start_t:end_t, :, line_idx] = 0

        # 3) 距离衰减 + 距离相关噪声：模拟远距离回波变弱且噪声更大
        if random.random() < 0.60:
            y_grid, x_grid = np.ogrid[:H, :W]
            center_y, center_x = (H - 1) / 2.0, (W - 1) / 2.0
            dist = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
            dist_norm = dist / max(float(dist.max()), 1e-6)

            atten_strength = random.uniform(0.20, 0.6)
            atten = np.exp(-atten_strength * (dist_norm ** 2)).astype(np.float32)
            tensor = tensor * atten[None, :, :]

            base_std = float(np.std(tensor)) + 1e-6
            noise_level = random.uniform(0.01, 0.2) * base_std
            range_sigma = (0.35 + 0.65 * dist_norm).astype(np.float32)
            noise = np.random.normal(0.0, 1.0, size=tensor.shape).astype(np.float32)
            signal_mask = (np.abs(tensor) > 1e-8).astype(np.float32)
            tensor = tensor + noise * (noise_level * range_sigma[None, :, :]) * signal_mask

        # 4) 无人机姿态微抖：逐帧小平移，边缘补零
        if random.random() < 0.70:
            max_shift = max(1, int(min(H, W) * 0.2))
            for t in range(T):
                dx = random.randint(-max_shift, max_shift)
                dy = random.randint(-max_shift, max_shift)
                if dx == 0 and dy == 0:
                    continue

                shifted = np.roll(tensor[t], shift=(dy, dx), axis=(0, 1))

                if dy > 0:
                    shifted[:dy, :] = 0
                elif dy < 0:
                    shifted[dy:, :] = 0
                if dx > 0:
                    shifted[:, :dx] = 0
                elif dx < 0:
                    shifted[:, dx:] = 0

                tensor[t] = shifted

        return tensor

    def __getitem__(self, idx):
        mat_path = self.file_list[idx]
        data = sio.loadmat(mat_path)

        # 原始输入: [Frames, H, W]
        tensor_raw = data['tensor'].astype(np.float32)

        # 训练时可选增强
        if self.augment:
            tensor_raw = self.apply_masking(tensor_raw)

        # 归一化
        tensor_norm = tensor_raw / self.cfg.lidar_scale
        input_tensor = torch.from_numpy(tensor_norm.astype(np.float32).copy()).unsqueeze(0)

        # 标签: 方向(sin, cos) + 波高(Hs)
        labels_raw = data['labels'].astype(np.float32).flatten()
        target_dir = torch.from_numpy(labels_raw[0:2])
        hs_value = labels_raw[2]
        hs_norm = hs_value / self.cfg.max_hs
        target_hs = torch.tensor([hs_norm], dtype=torch.float32)

        return input_tensor, target_dir, target_hs
