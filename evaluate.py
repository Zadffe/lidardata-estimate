import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from data.dataset import LidarWaveDataset
from models.wave_cnn import build_model


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def resolve_inference_checkpoint_path(cfg):
    explicit_path = str(getattr(cfg, "inference_checkpoint_path", "") or "").strip()
    if explicit_path:
        return explicit_path

    if getattr(cfg, "inference_use_latest_checkpoint", False):
        return os.path.join(cfg.save_dir, "latest_checkpoint.pth")

    return os.path.join(cfg.save_dir, "best_model.pth")


def load_inference_weights(model, checkpoint_path, device):
    """
    兼容两种权重文件:
    1. 纯模型权重: model.state_dict()
    2. 完整训练 checkpoint: 包含 model_state_dict 的字典
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return "full_checkpoint"

    model.load_state_dict(checkpoint)
    return "model_state_dict"


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


def evaluate_and_plot():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_inference_checkpoint_path(cfg)

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(cfg.log_dir, f"evaluate_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    print(f"评估设备: {device}")
    print(f"当前模型: {cfg.model_name}")
    print(f"权重文件: {checkpoint_path}")
    print(f"结果输出目录: {cfg.results_dir}")
    print(f"测试集增强: {cfg.test_augment}")

    logging.info(f"Evaluation started. Device: {device}")
    logging.info(f"Model name: {cfg.model_name}")
    logging.info(f"Checkpoint path: {checkpoint_path}")
    logging.info(f"Results directory: {cfg.results_dir}")
    logging.info(f"Test augmentation: {cfg.test_augment}")

    test_ds = LidarWaveDataset(mode="test", augment=cfg.test_augment)
    if len(test_ds) == 0:
        print("错误: 测试集为空，请检查 data_root 或数据生成流程。")
        return

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    model = build_model(cfg).to(device)
    if not os.path.exists(checkpoint_path):
        print(f"错误: 未找到权重文件 {checkpoint_path}")
        return

    checkpoint_type = load_inference_weights(model, checkpoint_path, device)
    logging.info(f"Checkpoint format: {checkpoint_type}")
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

    csv_timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(cfg.results_dir, f"evaluation_results_{csv_timestamp}.csv")
    df_results = pd.DataFrame(
        {
            "True_Hs": true_hs,
            "Pred_Hs": pred_hs,
            "True_Dir": true_dir,
            "Pred_Dir": pred_dir,
            "Hs_Error": pred_hs - true_hs,
            "Dir_Error_Abs": np.minimum(np.abs(true_dir - pred_dir), 360 - np.abs(true_dir - pred_dir)),
        }
    )
    df_results.to_csv(csv_path, index=False)
    logging.info(f"Detailed prediction results saved to: {csv_path}")
    print(f"详细结果已保存到: {csv_path}")

    r2_hs = r2_score(true_hs, pred_hs)
    rmse_hs = np.sqrt(mean_squared_error(true_hs, pred_hs))
    mae_hs = mean_absolute_error(true_hs, pred_hs)
    bias_hs = np.mean(pred_hs - true_hs)

    r2_dir = circular_r2_score(true_dir, pred_dir)
    diff_dir_abs = np.abs(true_dir - pred_dir)
    diff_dir_abs = np.minimum(diff_dir_abs, 360 - diff_dir_abs)
    rmse_dir = np.sqrt(np.mean(diff_dir_abs**2))
    mae_dir = np.mean(diff_dir_abs)

    diff_dir_signed = pred_dir - true_dir
    diff_dir_signed[diff_dir_signed > 180] -= 360
    diff_dir_signed[diff_dir_signed < -180] += 360
    bias_dir = np.mean(diff_dir_signed)

    logging.info("=== Detailed Evaluation Metrics ===")
    logging.info("[Wave Height Hs]")
    logging.info(f"  R2 Score: {r2_hs:.4f}")
    logging.info(f"  RMSE:     {rmse_hs:.4f} m")
    logging.info(f"  MAE:      {mae_hs:.4f} m")
    logging.info(f"  Bias:     {bias_hs:.4f} m")

    logging.info("[Wave Direction Dir]")
    logging.info(f"  Circ R2:  {r2_dir:.4f}")
    logging.info(f"  RMSE:     {rmse_dir:.4f} deg")
    logging.info(f"  MAE:      {mae_dir:.4f} deg")
    logging.info(f"  Bias:     {bias_dir:.4f} deg")

    print(f"\n波高 Hs: R2={r2_hs:.4f} | RMSE={rmse_hs:.4f} m | MAE={mae_hs:.4f} m | Bias={bias_hs:.4f} m")
    print(f"波向 Dir: R2={r2_dir:.4f} | RMSE={rmse_dir:.4f} deg | MAE={mae_dir:.4f} deg | Bias={bias_dir:.4f} deg")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    sns.scatterplot(x=true_hs, y=pred_hs, ax=ax, alpha=0.6, color="blue", edgecolor="k")
    lims = [0, 6]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction Line")
    ax.set_title(f"Wave height prediction analysis ($R^2$={r2_hs:.3f})", fontsize=14)
    ax.set_xlabel("Real Height (m)", fontsize=12)
    ax.set_ylabel("Predicted Height (m)", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    ax = axes[0, 1]
    error_hs = pred_hs - true_hs
    sns.histplot(error_hs, kde=True, ax=ax, color="green", bins=30)
    ax.axvline(0, color="r", linestyle="--")
    ax.set_title("Height Error Distribution", fontsize=14)
    ax.set_xlabel("Prediction Error (m)", fontsize=12)
    ax.set_ylabel("Sample size", fontsize=12)

    ax = axes[1, 0]
    sns.scatterplot(x=true_dir, y=pred_dir, ax=ax, alpha=0.6, color="purple", edgecolor="k")
    ax.plot([0, 360], [0, 360], "r--", linewidth=2)
    ax.set_title(f"Wave direction prediction analysis ($R^2$={r2_dir:.3f})", fontsize=14)
    ax.set_xlabel("Real angle (deg)", fontsize=12)
    ax.set_ylabel("Prediction angle (deg)", fontsize=12)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.grid(True)

    ax = axes[1, 1]
    diff = pred_dir - true_dir
    diff[diff > 180] -= 360
    diff[diff < -180] += 360
    sns.histplot(diff, kde=True, ax=ax, color="orange", bins=30)
    ax.axvline(0, color="r", linestyle="--")
    ax.set_title("Angular Error Distribution", fontsize=14)
    ax.set_xlabel("Angular Error (deg)", fontsize=12)
    ax.set_ylabel("Sample size", fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(cfg.results_dir, "scatter_plot.png")
    plt.savefig(save_path, dpi=300)
    logging.info(f"Plot saved to: {save_path}")
    print(f"图像已保存到: {save_path}")
    plt.show()


if __name__ == "__main__":
    evaluate_and_plot()
