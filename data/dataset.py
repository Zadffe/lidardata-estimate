# -*- coding: utf-8 -*-
import glob
import os
import random

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from configs.config import Config


class LidarWaveDataset(Dataset):
    def __init__(self, mode="train", augment=False, cfg=None):
        """
        Args:
            mode: "train" / "val" / "test"
            augment: 是否启用数据增强
            cfg: 运行时配置；如果不传，则回退到默认 Config()
        """

        self.cfg = cfg if cfg is not None else Config()
        self.mode = mode
        self.augment = augment
        self.file_list = sorted(glob.glob(os.path.join(self.cfg.data_root, mode, "*.mat")))

        if len(self.file_list) == 0:
            print(f"警告: {mode} 集未找到 .mat 文件")

        # 每一帧固定丢失 80% 的有效点，同时保留中心更密、边缘更稀的分布特性。
        self.frame_dropout_rate = 0.80

    def __len__(self):
        return len(self.file_list)

    def apply_range_dependent_dropout(self, tensor: np.ndarray) -> np.ndarray:
        """
        距离衰减：离中心越远，点云越容易丢失。
        输入/输出均为 [T, H, W]。
        """

        t_size, height, width = tensor.shape
        y_grid, x_grid = np.ogrid[:height, :width]
        center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
        dist = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
        dist_norm = dist / max(float(dist.max()), 1e-6)

        near_drop = random.uniform(0.02, 0.10)
        far_drop = random.uniform(0.35, 0.75)
        gamma = random.uniform(1.2, 2.4)
        drop_prob = near_drop + (far_drop - near_drop) * (dist_norm**gamma)
        keep_prob = 1.0 - drop_prob

        signal_mask = (np.abs(tensor) > 1e-8).astype(np.float32)
        random_keep = (np.random.rand(t_size, height, width) < keep_prob[None, :, :]).astype(np.float32)
        return tensor * signal_mask * random_keep

    def apply_framewise_density_jitter(self, tensor: np.ndarray) -> np.ndarray:
        """
        每一帧随机一个局部高密度区域，模拟无人机抖动导致的点云密度变化。
        """

        tensor = tensor.copy()
        t_size, height, width = tensor.shape

        y_coords = np.linspace(-(height - 1) / 2.0, (height - 1) / 2.0, height, dtype=np.float32)
        x_coords = np.linspace(-(width - 1) / 2.0, (width - 1) / 2.0, width, dtype=np.float32)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

        for t in range(t_size):
            hotspot_x = random.uniform(float(x_coords.min()) * 0.9, float(x_coords.max()) * 0.9)
            hotspot_y = random.uniform(float(y_coords.min()) * 0.9, float(y_coords.max()) * 0.9)
            sigma = random.uniform(max(4.0, min(height, width) * 0.08), max(8.0, min(height, width) * 0.22))

            dist2 = (xx - hotspot_x) ** 2 + (yy - hotspot_y) ** 2
            hotspot = np.exp(-dist2 / (2.0 * sigma**2)).astype(np.float32)

            base_keep = random.uniform(0.08, 0.25)
            hotspot_gain = random.uniform(0.55, 0.85)
            keep_prob = np.clip(base_keep + hotspot_gain * hotspot, 0.02, 0.98)

            signal_mask = (np.abs(tensor[t]) > 1e-8).astype(np.float32)
            random_keep = (np.random.rand(height, width) < keep_prob).astype(np.float32)
            tensor[t] = tensor[t] * signal_mask * random_keep

        return tensor

    def apply_block_dropout(self, tensor: np.ndarray) -> np.ndarray:
        """
        局部块状缺失：随机若干帧或一小段时间内，某个连续区域直接丢失。
        """

        tensor = tensor.copy()
        t_size, height, width = tensor.shape

        n_blocks = random.randint(1, 3)
        for _ in range(n_blocks):
            block_h = random.randint(5, max(5, min(height // 8, 12)))
            block_w = random.randint(5, max(5, min(width // 8, 12)))

            top = random.randint(0, max(0, height - block_h))
            left = random.randint(0, max(0, width - block_w))

            start_t = random.randint(0, t_size - 1)
            duration = random.randint(1, max(1, t_size // 6))
            end_t = min(t_size, start_t + duration)

            tensor[start_t:end_t, top : top + block_h, left : left + block_w] = 0

        return tensor

    def apply_fixed_frame_dropout(self, tensor: np.ndarray) -> np.ndarray:
        """
        对每一帧执行固定比例丢点，同时引入中心先验和局部热点。
        输入/输出均为 [T, H, W]。
        """

        tensor = tensor.copy()
        t_size, height, width = tensor.shape
        keep_ratio = 1.0 - self.frame_dropout_rate
        y_coords = np.arange(height, dtype=np.float32)
        x_coords = np.arange(width, dtype=np.float32)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0

        dist_to_center = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        dist_norm = dist_to_center / max(float(dist_to_center.max()), 1e-6)

        for t in range(t_size):
            valid_mask = np.abs(tensor[t]) > 1e-8
            valid_idx = np.flatnonzero(valid_mask)
            valid_count = valid_idx.size
            if valid_count == 0:
                continue

            keep_count = max(1, int(round(valid_count * keep_ratio)))
            if keep_count >= valid_count:
                continue

            hotspot_radius_x = width * 0.18
            hotspot_radius_y = height * 0.18
            hotspot_x = center_x + random.uniform(-hotspot_radius_x, hotspot_radius_x)
            hotspot_y = center_y + random.uniform(-hotspot_radius_y, hotspot_radius_y)
            sigma_x = random.uniform(max(3.0, width * 0.10), max(6.0, width * 0.24))
            sigma_y = random.uniform(max(3.0, height * 0.10), max(6.0, height * 0.24))
            theta = random.uniform(0.0, np.pi)

            x_shift = xx - hotspot_x
            y_shift = yy - hotspot_y
            x_rot = x_shift * np.cos(theta) + y_shift * np.sin(theta)
            y_rot = -x_shift * np.sin(theta) + y_shift * np.cos(theta)

            hotspot = np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)).astype(np.float32)

            center_gamma = random.uniform(1.6, 2.4)
            center_floor = random.uniform(0.02, 0.06)
            center_prior = center_floor + (1.0 - center_floor) * (1.0 - dist_norm**center_gamma)
            center_prior = np.clip(center_prior, center_floor, 1.0).astype(np.float32)

            baseline = random.uniform(0.03, 0.08)
            hotspot_gain = random.uniform(0.88, 1.15)
            weight_map = center_prior * (baseline + hotspot_gain * hotspot)
            weight_map = weight_map + np.random.rand(height, width).astype(np.float32) * 1e-3

            valid_weights = weight_map.reshape(-1)[valid_idx]
            valid_weights = valid_weights / np.sum(valid_weights)
            keep_idx = np.random.choice(valid_idx, size=keep_count, replace=False, p=valid_weights)

            frame_mask = np.zeros(tensor[t].size, dtype=bool)
            frame_mask[keep_idx] = True
            tensor[t] = tensor[t] * frame_mask.reshape(tensor[t].shape)

        return tensor

    def apply_masking(self, tensor: np.ndarray) -> np.ndarray:
        return self.apply_fixed_frame_dropout(tensor)

    def __getitem__(self, idx):
        mat_path = self.file_list[idx]
        data = sio.loadmat(mat_path)

        tensor_raw = data["tensor"].astype(np.float32)
        if self.augment:
            tensor_raw = self.apply_masking(tensor_raw)

        tensor_norm = tensor_raw / self.cfg.lidar_scale
        input_tensor = torch.from_numpy(tensor_norm.astype(np.float32).copy()).unsqueeze(0)

        labels_raw = data["labels"].astype(np.float32).flatten()
        target_dir = torch.from_numpy(labels_raw[0:2])
        hs_value = labels_raw[2]
        hs_norm = hs_value / self.cfg.max_hs
        target_hs = torch.tensor([hs_norm], dtype=torch.float32)

        return input_tensor, target_dir, target_hs
