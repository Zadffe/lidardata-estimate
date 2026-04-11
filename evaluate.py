import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import LidarWaveDataset
from models.wave_cnn import build_model
from utils.checkpoint_utils import load_model_weights
from utils.runtime_utils import (
    add_common_eval_args,
    build_eval_cfg_from_args,
    resolve_device,
    setup_logging,
)


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def circular_r2_score(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    diff = np.minimum(diff, 360 - diff)
    ss_res = np.sum(diff**2)

    y_true_rad = np.deg2rad(y_true)
    sin_sum = np.sum(np.sin(y_true_rad))
    cos_sum = np.sum(np.cos(y_true_rad))
    mean_angle = np.degrees(np.arctan2(sin_sum, cos_sum))
    if mean_angle < 0:
        mean_angle += 360

    diff_mean = np.abs(y_true - mean_angle)
    diff_mean = np.minimum(diff_mean, 360 - diff_mean)
    ss_tot = np.sum(diff_mean**2)

    if ss_tot == 0:
        return 0.0

    return 1 - ss_res / ss_tot


def create_parser():
    parser = argparse.ArgumentParser(
        description="已训练模型评估脚本。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_eval_args(parser)
    return parser


def make_summary_figure(true_hs, pred_hs, true_dir, pred_dir, metrics, save_path):
    diff_dir_signed = pred_dir - true_dir
    diff_dir_signed[diff_dir_signed > 180] -= 360
    diff_dir_signed[diff_dir_signed < -180] += 360
    diff_dir_abs = np.abs(diff_dir_signed)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=180)
    fig.suptitle("Wave Height & Direction Evaluation Summary", fontsize=18, y=1.02)

    ax = axes[0, 0]
    ax.scatter(true_hs, pred_hs, s=22, alpha=0.65, c="#1f77b4", edgecolors="none")
    lims = [min(true_hs.min(), pred_hs.min()), max(true_hs.max(), pred_hs.max())]
    ax.plot(lims, lims, "--", color="#d62728", linewidth=2)
    ax.set_title("Hs Scatter")
    ax.set_xlabel("True Hs (m)")
    ax.set_ylabel("Predicted Hs (m)")
    ax.text(
        0.03,
        0.97,
        f"R²={metrics['r2_hs']:.3f}\nRMSE={metrics['rmse_hs']:.3f} m\nMAE={metrics['mae_hs']:.3f} m",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    ax = axes[0, 1]
    error_hs = pred_hs - true_hs
    sns.histplot(error_hs, kde=True, bins=30, color="#2ca02c", ax=ax)
    ax.axvline(0, color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_title("Hs Residual Distribution")
    ax.set_xlabel("Predicted - True (m)")
    ax.set_ylabel("Count")

    ax = axes[0, 2]
    ax.scatter(true_hs, error_hs, s=20, alpha=0.6, c="#17becf", edgecolors="none")
    ax.axhline(0, color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_title("Hs Residual vs True Hs")
    ax.set_xlabel("True Hs (m)")
    ax.set_ylabel("Residual (m)")

    ax = axes[1, 0]
    ax.scatter(true_dir, pred_dir, s=20, alpha=0.65, c="#9467bd", edgecolors="none")
    ax.plot([0, 360], [0, 360], "--", color="#d62728", linewidth=2)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_title("Direction Scatter")
    ax.set_xlabel("True Direction (deg)")
    ax.set_ylabel("Predicted Direction (deg)")
    ax.text(
        0.03,
        0.97,
        f"R²={metrics['r2_dir']:.3f}\nRMSE={metrics['rmse_dir']:.3f}°\nMAE={metrics['mae_dir']:.3f}°",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    ax = axes[1, 1]
    sns.histplot(diff_dir_signed, kde=True, bins=36, color="#ff7f0e", ax=ax)
    ax.axvline(0, color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_title("Signed Angular Error")
    ax.set_xlabel("Predicted - True (deg)")
    ax.set_ylabel("Count")

    ax = axes[1, 2]
    thresholds = np.arange(0, 31, 1)
    coverage = [(diff_dir_abs <= t).mean() * 100 for t in thresholds]
    ax.plot(thresholds, coverage, color="#8c564b", linewidth=2.5)
    ax.set_title("Angular Error Coverage")
    ax.set_xlabel("Absolute Error Threshold (deg)")
    ax.set_ylabel("Samples Within Threshold (%)")
    ax.set_xlim(0, thresholds.max())
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.text(
        0.03,
        0.97,
        f"<=5° : {(diff_dir_abs <= 5).mean() * 100:.1f}%\n"
        f"<=10°: {(diff_dir_abs <= 10).mean() * 100:.1f}%\n"
        f"<=15°: {(diff_dir_abs <= 15).mean() * 100:.1f}%",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_and_plot(args):
    cfg = build_eval_cfg_from_args(args)
    device = resolve_device(cfg.device)

    os.makedirs(cfg.results_dir, exist_ok=True)
    logger, log_file = setup_logging(cfg.log_dir, "evaluate")

    print(f"评估设备: {device}")
    print(f"当前模型: {cfg.model_name}")
    print(f"权重文件: {cfg.checkpoint_path}")
    print(f"结果输出目录: {cfg.results_dir}")
    print(f"日志文件: {log_file}")
    print(f"测试集增强: {cfg.test_augment}")

    logger.info(f"Evaluation started. Device: {device}")
    logger.info(f"Model name: {cfg.model_name}")
    logger.info(f"Checkpoint path: {cfg.checkpoint_path}")
    logger.info(f"Results directory: {cfg.results_dir}")
    logger.info(f"Test augmentation: {cfg.test_augment}")

    test_ds = LidarWaveDataset(mode="test", augment=cfg.test_augment, cfg=cfg)
    if len(test_ds) == 0:
        raise RuntimeError("测试集为空，请检查 --data-root 路径和数据集目录结构。")

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    model = build_model(cfg).to(device)
    if not os.path.exists(cfg.checkpoint_path):
        raise FileNotFoundError(f"未找到权重文件: {cfg.checkpoint_path}")

    _, checkpoint_type = load_model_weights(model, cfg.checkpoint_path, device)
    logger.info(f"Checkpoint format: {checkpoint_type}")
    model.eval()

    true_hs_list, pred_hs_list = [], []
    true_dir_list, pred_dir_list = [], []

    with torch.no_grad():
        for inputs, target_dir, target_hs in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device, non_blocking=True)
            pred_dir_vec, pred_hs_norm = model(inputs)

            pred_hs = pred_hs_norm.cpu().numpy().flatten() * cfg.max_hs
            true_hs = target_hs.numpy().flatten() * cfg.max_hs
            true_hs_list.extend(true_hs)
            pred_hs_list.extend(pred_hs)

            p_sin = pred_dir_vec[:, 0].cpu().numpy()
            p_cos = pred_dir_vec[:, 1].cpu().numpy()
            p_ang = np.degrees(np.arctan2(p_sin, p_cos))
            p_ang[p_ang < 0] += 360

            t_sin = target_dir[:, 0].numpy()
            t_cos = target_dir[:, 1].numpy()
            t_ang = np.degrees(np.arctan2(t_sin, t_cos))
            t_ang[t_ang < 0] += 360

            true_dir_list.extend(t_ang)
            pred_dir_list.extend(p_ang)

    true_hs = np.array(true_hs_list)
    pred_hs = np.array(pred_hs_list)
    true_dir = np.array(true_dir_list)
    pred_dir = np.array(pred_dir_list)

    diff_dir_abs = np.abs(true_dir - pred_dir)
    diff_dir_abs = np.minimum(diff_dir_abs, 360 - diff_dir_abs)
    diff_dir_signed = pred_dir - true_dir
    diff_dir_signed[diff_dir_signed > 180] -= 360
    diff_dir_signed[diff_dir_signed < -180] += 360

    metrics = {
        "r2_hs": r2_score(true_hs, pred_hs),
        "rmse_hs": np.sqrt(mean_squared_error(true_hs, pred_hs)),
        "mae_hs": mean_absolute_error(true_hs, pred_hs),
        "bias_hs": np.mean(pred_hs - true_hs),
        "r2_dir": circular_r2_score(true_dir, pred_dir),
        "rmse_dir": np.sqrt(np.mean(diff_dir_abs**2)),
        "mae_dir": np.mean(diff_dir_abs),
        "bias_dir": np.mean(diff_dir_signed),
    }

    checkpoint_stem = os.path.basename(cfg.checkpoint_path).replace(".pth", "")
    csv_path = os.path.join(cfg.results_dir, f"evaluation_results_{checkpoint_stem}.csv")
    metrics_path = os.path.join(cfg.results_dir, f"metrics_{checkpoint_stem}.csv")
    figure_path = os.path.join(cfg.results_dir, f"evaluation_summary_{checkpoint_stem}.png")

    df_results = pd.DataFrame(
        {
            "True_Hs": true_hs,
            "Pred_Hs": pred_hs,
            "Hs_Error": pred_hs - true_hs,
            "True_Dir": true_dir,
            "Pred_Dir": pred_dir,
            "Dir_Error_Signed": diff_dir_signed,
            "Dir_Error_Abs": diff_dir_abs,
        }
    )
    df_results.to_csv(csv_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    make_summary_figure(true_hs, pred_hs, true_dir, pred_dir, metrics, figure_path)

    logger.info(f"Detailed prediction results saved to: {csv_path}")
    logger.info(f"Metrics saved to: {metrics_path}")
    logger.info(f"Figure saved to: {figure_path}")

    print(f"\n波高 Hs: R2={metrics['r2_hs']:.4f} | RMSE={metrics['rmse_hs']:.4f} m | MAE={metrics['mae_hs']:.4f} m | Bias={metrics['bias_hs']:.4f} m")
    print(f"波向 Dir: R2={metrics['r2_dir']:.4f} | RMSE={metrics['rmse_dir']:.4f} deg | MAE={metrics['mae_dir']:.4f} deg | Bias={metrics['bias_dir']:.4f} deg")
    print(f"详细结果已保存到: {csv_path}")
    print(f"指标汇总已保存到: {metrics_path}")
    print(f"图像已保存到: {figure_path}")


def main():
    parser = create_parser()
    args = parser.parse_args()
    evaluate_and_plot(args)


if __name__ == "__main__":
    main()
