"""
visualize_augmentation.py
---------------------------
从 train / val / test 中随机取一个样本，
对同一份原始点云做一次数据增强，
并排可视化「原始帧」与「增强后帧」，方便你直观确认增强效果。

用法:
    python visualize_augmentation.py                  # 随机取一个 test 样本，展示中间那帧
    python visualize_augmentation.py --mode train --frame 10   # 指定 mode 和帧索引
    python visualize_augmentation.py --mode train --sample 3   # 指定样本索引
"""

import argparse
import os
import random
import glob
import math

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  触发 3D 投影注册

# ── 中文字体配置（Windows 优先 SimHei，找不到则用系统第一个支持 CJK 的字体）──
def _setup_chinese_font():
    preferred = ['SimHei', 'Microsoft YaHei', 'STSong', 'STHeiti', 'WenQuanYi Micro Hei']
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False
            return
    # 兜底：查找系统里任意含中文的字体文件
    cjk_paths = [f.fname for f in fm.fontManager.ttflist
                 if any(k in f.fname for k in ('Hei', 'Song', 'CJK', 'SC', 'TC', 'Gothic'))]
    if cjk_paths:
        plt.rcParams['font.family'] = fm.FontProperties(fname=cjk_paths[0]).get_name()
        plt.rcParams['axes.unicode_minus'] = False

_setup_chinese_font()

from configs.config import Config


# ─────────────────────────────────────────────
# 独立的 apply_masking（和 dataset.py 完全一致）
# ─────────────────────────────────────────────
def apply_masking(tensor: np.ndarray) -> np.ndarray:
    """Augment [T, H, W] tensor with UAV LiDAR-like artifacts."""
    tensor = tensor.copy()
    T, H, W = tensor.shape

    # 1) Dynamic random dropout: random missing returns on water surface.
    if random.random() < 0.8:
        drop_prob = random.uniform(0.2, 0.9)
        rand_mask = np.random.choice([0, 1], size=(T, H, W), p=[drop_prob, 1 - drop_prob])
        tensor = tensor * rand_mask

    # 2) Scan-line packet loss: contiguous rows/cols disappear for short time.
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

    # 3) Range attenuation + range-dependent noise.
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

    # 4) UAV pose jitter: frame-wise small translation with blank edges.
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


# ─────────────────────────────────────────────
# 工具：解析标签角度
# ─────────────────────────────────────────────
def labels_to_readable(labels: np.ndarray, max_hs: float) -> dict:
    sin_val, cos_val = float(labels[0]), float(labels[1])
    angle_deg = math.degrees(math.atan2(sin_val, cos_val))
    if angle_deg < 0:
        angle_deg += 360.0
    hs = float(labels[2])
    return {"dir_deg": angle_deg, "sin": sin_val, "cos": cos_val, "hs": hs}


# ─────────────────────────────────────────────
# 主可视化函数
# ─────────────────────────────────────────────
def visualize(mode: str = 'test', sample_idx: int = None, frame_idx: int = None,
              save_path: str = None, min_abs_value: float = 0.0, z_scale: float = 1.0):
    cfg = Config()
    file_list = sorted(glob.glob(os.path.join(cfg.data_root, mode, '*.mat')))

    if len(file_list) == 0:
        print(f"[Error] No .mat files found in: {os.path.join(cfg.data_root, mode)}")
        return

    if sample_idx is None:
        sample_idx = random.randint(0, len(file_list) - 1)
    sample_idx = sample_idx % len(file_list)
    mat_path = file_list[sample_idx]

    data = sio.loadmat(mat_path)
    tensor_raw = data['tensor'].astype(np.float32)
    labels_raw = data['labels'].astype(np.float32).flatten()
    T, H, W = tensor_raw.shape

    if frame_idx is None:
        frame_idx = T // 2
    frame_idx = max(0, min(frame_idx, T - 1))

    tensor_aug = apply_masking(tensor_raw)
    frame_orig = tensor_raw[frame_idx]
    frame_aug = tensor_aug[frame_idx]

    info = labels_to_readable(labels_raw, cfg.max_hs)
    sample_name = os.path.basename(mat_path)

    point_threshold = max(float(min_abs_value), 1e-8)

    def point_mask(frame: np.ndarray):
        # Treat near-zero as missing points for visualization/drop statistics.
        return np.isfinite(frame) & (np.abs(frame) > point_threshold)

    def frame_to_3d(frame: np.ndarray):
        valid_mask = point_mask(frame)
        ys, xs = np.where(valid_mask)
        zs = frame[ys, xs]
        return xs.astype(float), ys.astype(float), zs.astype(float)

    def calc_z_visual_params(frames):
        z_values = []
        for frame in frames:
            mask = point_mask(frame)
            if np.any(mask):
                z_values.append(frame[mask].astype(float))

        if not z_values:
            return 1.0, 0.0, (-1.0, 1.0)

        z_all = np.concatenate(z_values)
        z_center = float(np.median(z_all))
        z_range = max(float(np.ptp(z_all)), 1e-6)

        xy_span = max(float(H - 1), float(W - 1), 1.0)
        auto_scale = float(np.clip(xy_span / z_range, 1.0, 4000.0))
        z_vis_scale = auto_scale * max(float(z_scale), 1e-3)

        z_vis_all = (z_all - z_center) * z_vis_scale + z_center
        z_pad = max(float(np.ptp(z_vis_all)) * 0.08, 1e-6)
        z_vis_lim = (float(np.min(z_vis_all) - z_pad), float(np.max(z_vis_all) + z_pad))

        return z_vis_scale, z_center, z_vis_lim

    z_vis_scale, z_vis_center, z_vis_lim = calc_z_visual_params([frame_orig, frame_aug])

    def to_visual_z(z_values):
        return (z_values - z_vis_center) * z_vis_scale + z_vis_center

    def draw_3d_cloud(ax3d, frame, title, color_by_z=True, cmap='viridis'):
        xs, ys, zs = frame_to_3d(frame)

        if len(zs) == 0:
            ax3d.set_title(title + "\n(no valid points)", fontsize=11)
            return

        zs_vis = to_visual_z(zs)

        sc = ax3d.scatter(
            xs, ys, zs_vis,
            c=zs if color_by_z else 'steelblue',
            cmap=cmap,
            vmin=float(np.nanmin(frame)), vmax=float(np.nanmax(frame)),
            s=1.2,
            alpha=0.45,
            depthshade=False
        )
        plt.colorbar(sc, ax=ax3d, shrink=0.5, pad=0.05, label='Height (raw)')

        cx, cy = W / 2.0, H / 2.0
        center_z_vis = float(to_visual_z(np.array([0.0]))[0])
        ax3d.scatter([cx], [cy], [center_z_vis], color='red', s=60, zorder=5, label='LiDAR center')

        x_range = max(float(np.ptp(xs)), 1e-6)
        y_range = max(float(np.ptp(ys)), 1e-6)
        z_range_vis = max(float(np.ptp(zs_vis)), 1e-6)

        ax3d.set_box_aspect((x_range, y_range, z_range_vis))
        ax3d.set_proj_type('ortho')
        ax3d.set_xlim(0, max(W - 1, 1))
        ax3d.set_ylim(0, max(H - 1, 1))
        ax3d.set_zlim(*z_vis_lim)

        ax3d.set_xlabel('X [pixel]', labelpad=6)
        ax3d.set_ylabel('Y [pixel]', labelpad=6)
        ax3d.set_zlabel(f'Height (display x{z_vis_scale:.1f})', labelpad=6)
        ax3d.set_title(title, fontsize=11)
        ax3d.legend(fontsize=7, loc='upper left')
        ax3d.view_init(elev=30, azim=-60)

    orig_mask = point_mask(frame_orig)
    aug_mask = point_mask(frame_aug)
    valid_orig = np.count_nonzero(orig_mask)
    valid_aug = np.count_nonzero(aug_mask)
    dropped_cnt = np.count_nonzero(orig_mask & (~aug_mask))
    drop_rate = (dropped_cnt / valid_orig) if valid_orig > 0 else 0.0

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        f"Sample: {sample_name} | Frame: {frame_idx}/{T - 1}\n"
        f"Dir: {info['dir_deg']:.1f} deg | Hs: {info['hs']:.2f} m | "
        f"Drop rate ~= {drop_rate * 100:.1f}% | Z display x{z_vis_scale:.1f}",
        fontsize=12
    )

    ax_orig = fig.add_subplot(1, 2, 1, projection='3d')
    draw_3d_cloud(ax_orig, frame_orig, title=f"Original point cloud\nvalid: {valid_orig}")

    ax_aug = fig.add_subplot(1, 2, 2, projection='3d')
    draw_3d_cloud(ax_aug, frame_aug, title=f"Augmented point cloud\nvalid: {valid_aug} (drop ~= {drop_rate * 100:.1f}%)")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    plt.show()

    print(f"\nSample path  : {mat_path}")
    print(f"Frame index  : {frame_idx} / {T - 1}")
    print(f"Direction    : {info['dir_deg']:.1f} deg")
    print(f"Hs           : {info['hs']:.2f} m")
    print(f"Drop rate    : {drop_rate * 100:.1f}%")
    print(f"Z vis factor : {z_vis_scale:.1f}x (auto * --z_scale)")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化数据增强前后的雷达点云帧')
    parser.add_argument('--mode',   type=str, default='test',
                        help="数据集分割: train / val / test  (默认 test)")
    parser.add_argument('--sample', type=int, default=None,
                        help="样本索引（不填则随机选）")
    parser.add_argument('--frame',  type=int, default=None,
                        help="帧索引（不填则取中间帧）")
    parser.add_argument('--save',   type=str, default=None,
                        help="可选：将图像保存到此路径，如 results/aug_vis.png")
    parser.add_argument('--min_abs', type=float, default=0.0,
                        help="最小绝对值阈值，默认0.0表示显示所有有效点（含负值）")
    parser.add_argument('--z_scale', type=float, default=0.3,
                        help="Additional multiplier for adaptive Z stretching (default: 1.0).")
    args = parser.parse_args()

    visualize(
        mode=args.mode,
        sample_idx=args.sample,
        frame_idx=args.frame,
        save_path=args.save,
        min_abs_value=args.min_abs,
        z_scale=args.z_scale
    )

