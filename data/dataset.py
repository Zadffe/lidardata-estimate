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

        # 每一帧固定丢失 80% 的有效点，但高密度区域只会出现在中心附近。
        self.frame_dropout_rate = 0.80

    def __len__(self):
        return len(self.file_list)

    def apply_range_dependent_dropout(self, tensor: np.ndarray) -> np.ndarray:
        """
        距离衰减：离雷达中心越远，点云越容易丢失。
        输入/输出均为 [T, H, W]。
        """
        T, H, W = tensor.shape
        y_grid, x_grid = np.ogrid[:H, :W]
        center_y, center_x = (H - 1) / 2.0, (W - 1) / 2.0
        dist = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
        dist_norm = dist / max(float(dist.max()), 1e-6)

        near_drop = random.uniform(0.02, 0.10)
        far_drop = random.uniform(0.35, 0.75)
        gamma = random.uniform(1.2, 2.4)
        drop_prob = near_drop + (far_drop - near_drop) * (dist_norm ** gamma)
        keep_prob = 1.0 - drop_prob

        signal_mask = (np.abs(tensor) > 1e-8).astype(np.float32)
        random_keep = (np.random.rand(T, H, W) < keep_prob[None, :, :]).astype(np.float32)
        return tensor * signal_mask * random_keep

    def apply_framewise_density_jitter(self, tensor: np.ndarray) -> np.ndarray:
        """
        每一帧随机一个局部高密度区域，模拟无人机抖动导致的点云密度分布变化。
        某一帧可能左上更密，下一帧可能右下更密。
        """
        tensor = tensor.copy()
        T, H, W = tensor.shape

        y_coords = np.linspace(-(H - 1) / 2.0, (H - 1) / 2.0, H, dtype=np.float32)
        x_coords = np.linspace(-(W - 1) / 2.0, (W - 1) / 2.0, W, dtype=np.float32)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

        for t in range(T):
            hotspot_x = random.uniform(float(x_coords.min()) * 0.9, float(x_coords.max()) * 0.9)
            hotspot_y = random.uniform(float(y_coords.min()) * 0.9, float(y_coords.max()) * 0.9)
            sigma = random.uniform(max(4.0, min(H, W) * 0.08), max(8.0, min(H, W) * 0.22))

            dist2 = (xx - hotspot_x) ** 2 + (yy - hotspot_y) ** 2
            hotspot = np.exp(-dist2 / (2.0 * sigma ** 2)).astype(np.float32)

            base_keep = random.uniform(0.08, 0.25)
            hotspot_gain = random.uniform(0.55, 0.85)
            keep_prob = np.clip(base_keep + hotspot_gain * hotspot, 0.02, 0.98)

            signal_mask = (np.abs(tensor[t]) > 1e-8).astype(np.float32)
            random_keep = (np.random.rand(H, W) < keep_prob).astype(np.float32)
            tensor[t] = tensor[t] * signal_mask * random_keep

        return tensor

    def apply_block_dropout(self, tensor: np.ndarray) -> np.ndarray:
        """
        局部块状缺失：随机若干帧或一小段时间内，某个连续区域直接丢失。
        """
        tensor = tensor.copy()
        T, H, W = tensor.shape

        n_blocks = random.randint(1, 3)
        for _ in range(n_blocks):
            block_h = random.randint(5, max(5, min(H // 8, 12)))
            block_w = random.randint(5, max(5, min(W // 8, 12)))

            top = random.randint(0, max(0, H - block_h))
            left = random.randint(0, max(0, W - block_w))

            start_t = random.randint(0, T - 1)
            duration = random.randint(1, max(1, T // 6))
            end_t = min(T, start_t + duration)

            tensor[start_t:end_t, top:top + block_h, left:left + block_w] = 0

        return tensor

    def apply_fixed_frame_dropout(self, tensor: np.ndarray) -> np.ndarray:
        """
        对每一帧的有效点执行固定比例丢点，同时引入局部高密度热点。
        热点只在中心附近随机漂移，越靠近边缘保留概率越低，
        因而始终满足“中心更密、边缘更稀”的雷达衰减特性。
        输入/输出均为 [T, H, W]。
        """
        tensor = tensor.copy()
        T, H, W = tensor.shape
        keep_ratio = 1.0 - self.frame_dropout_rate
        y_coords = np.arange(H, dtype=np.float32)
        x_coords = np.arange(W, dtype=np.float32)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        center_x = (W - 1) / 2.0
        center_y = (H - 1) / 2.0

        dist_to_center = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        dist_norm = dist_to_center / max(float(dist_to_center.max()), 1e-6)

        for t in range(T):
            valid_mask = np.abs(tensor[t]) > 1e-8
            valid_idx = np.flatnonzero(valid_mask)
            valid_count = valid_idx.size
            if valid_count == 0:
                continue

            keep_count = max(1, int(round(valid_count * keep_ratio)))
            if keep_count >= valid_count:
                continue

            hotspot_radius_x = W * 0.18
            hotspot_radius_y = H * 0.18
            hotspot_x = center_x + random.uniform(-hotspot_radius_x, hotspot_radius_x)
            hotspot_y = center_y + random.uniform(-hotspot_radius_y, hotspot_radius_y)
            sigma_x = random.uniform(max(3.0, W * 0.10), max(6.0, W * 0.24))
            sigma_y = random.uniform(max(3.0, H * 0.10), max(6.0, H * 0.24))
            theta = random.uniform(0.0, np.pi)

            x_shift = xx - hotspot_x
            y_shift = yy - hotspot_y
            x_rot = x_shift * np.cos(theta) + y_shift * np.sin(theta)
            y_rot = -x_shift * np.sin(theta) + y_shift * np.cos(theta)

            hotspot = np.exp(
                -0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)
            ).astype(np.float32)

            center_gamma = random.uniform(1.6, 2.4)
            center_floor = random.uniform(0.02, 0.06)
            center_prior = center_floor + (1.0 - center_floor) * (1.0 - dist_norm ** center_gamma)
            center_prior = np.clip(center_prior, center_floor, 1.0).astype(np.float32)

            baseline = random.uniform(0.03, 0.08)
            hotspot_gain = random.uniform(0.88, 1.15)
            weight_map = center_prior * (baseline + hotspot_gain * hotspot)
            weight_map = weight_map + np.random.rand(H, W).astype(np.float32) * 1e-3

            valid_weights = weight_map.reshape(-1)[valid_idx]
            valid_weights = valid_weights / np.sum(valid_weights)
            keep_idx = np.random.choice(valid_idx, size=keep_count, replace=False, p=valid_weights)

            frame_mask = np.zeros(tensor[t].size, dtype=bool)
            frame_mask[keep_idx] = True
            tensor[t] = tensor[t] * frame_mask.reshape(tensor[t].shape)

        return tensor

    def apply_masking(self, tensor: np.ndarray) -> np.ndarray:
        """
        输入: [T, H, W] numpy 张量
        输出: 增强后的 numpy 张量
        """
        return self.apply_fixed_frame_dropout(tensor)

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
