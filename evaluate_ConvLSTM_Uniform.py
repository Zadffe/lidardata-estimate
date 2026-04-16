import argparse
import logging
import os
import sys
import time
from types import SimpleNamespace

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
from models.CNN_ConvLSTM_Uniform import build_model
from utils.checkpoint_utils import load_model_weights


DEFAULT_UNIFORM_DATA_ROOT = r"F:\Research__dir\dl_lidar\datasets\Dataset_Wave_Lidar_10000samples_uniform_grid"


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def setup_logging(log_dir, prefix):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger(prefix), log_file


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_default_experiment_name():
    return "ConvLSTM_uniform_grid_default"


def build_experiment_dirs(output_root, experiment_name):
    exp_root = os.path.join(output_root, experiment_name)
    return (
        exp_root,
        os.path.join(exp_root, "checkpoints"),
        os.path.join(exp_root, "results"),
        os.path.join(exp_root, "logs"),
    )


def infer_experiment_layout_from_checkpoint(checkpoint_path):
    checkpoint_path = os.path.abspath(str(checkpoint_path or "").strip())
    if not checkpoint_path:
        return None

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if os.path.basename(checkpoint_dir).lower() != "checkpoints":
        return None

    exp_root = os.path.dirname(checkpoint_dir)
    return {
        "experiment_name": os.path.basename(exp_root),
        "save_dir": checkpoint_dir,
        "results_dir": os.path.join(exp_root, "results"),
        "log_dir": os.path.join(exp_root, "logs"),
    }


def build_eval_cfg_from_args(args):
    base_cfg = Config()
    checkpoint_path = str(args.checkpoint_path or "").strip()
    checkpoint_layout = infer_experiment_layout_from_checkpoint(checkpoint_path) if checkpoint_path else None

    if checkpoint_layout:
        experiment_name = str(args.experiment_name or "").strip() or checkpoint_layout["experiment_name"]
        save_dir = checkpoint_layout["save_dir"]
        default_results_dir = checkpoint_layout["results_dir"]
        log_dir = checkpoint_layout["log_dir"]
        checkpoint_path = os.path.abspath(checkpoint_path)
    else:
        experiment_name = str(args.experiment_name or "").strip() or get_default_experiment_name()
        _, save_dir, default_results_dir, log_dir = build_experiment_dirs(args.output_root, experiment_name)

        if not checkpoint_path:
            checkpoint_name = str(args.checkpoint_file or "").strip()
            if not checkpoint_name:
                checkpoint_name = "latest_checkpoint.pth" if args.use_latest_checkpoint else "best_model.pth"
            checkpoint_path = os.path.join(save_dir, checkpoint_name)

    output_dir = str(args.output_dir or "").strip() or default_results_dir

    return SimpleNamespace(
        model_name="ConvLSTMUniformGrid",
        output_root=args.output_root,
        experiment_name=experiment_name,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        frames=base_cfg.frames,
        height=base_cfg.height,
        width=base_cfg.width,
        lidar_scale=base_cfg.lidar_scale,
        max_hs=base_cfg.max_hs,
        frame_dropout_rate=args.frame_dropout_rate,
        convlstm_hidden=args.convlstm_hidden,
        convlstm_layers=args.convlstm_layers,
        temporal_pool=args.temporal_pool,
        temporal_stride=args.temporal_stride,
        save_dir=save_dir,
        results_dir=output_dir,
        log_dir=log_dir,
        test_augment=args.test_augment,
        checkpoint_path=checkpoint_path,
        checkpoint_file=str(args.checkpoint_file or "").strip(),
        device=args.device,
        inference_use_latest_checkpoint=args.use_latest_checkpoint,
    )


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
        description="Evaluate the ConvLSTM uniform-grid model on the test split and save CSV metrics plus summary figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate_ConvLSTM_Uniform.py --experiment-name ConvLSTM_uniform_dataloss0.8_v1\n"
            "  python evaluate_ConvLSTM_Uniform.py --experiment-name ConvLSTM_uniform_dataloss0.8_v1 --checkpoint-file epoch_100.pth\n"
            "  python evaluate_ConvLSTM_Uniform.py --checkpoint-path ./all_exps_result/ConvLSTM_uniform_dataloss0.8_v1/checkpoints/best_model.pth"
        ),
    )
    parser.add_argument("--output-root", default=Config.output_root, help="Root directory that stores all experiment folders.")
    parser.add_argument("--experiment-name", default="", help="Experiment directory name under output root.")
    parser.add_argument("--checkpoint-path", default="", help="Explicit checkpoint path. Overrides experiment-based checkpoint lookup.")
    parser.add_argument(
        "--checkpoint-file",
        default="",
        help="Checkpoint filename inside the experiment checkpoints directory, for example best_model.pth or epoch_100.pth.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory used to store evaluation outputs. Defaults to the experiment results directory.",
    )
    parser.add_argument("--num-workers", type=int, default=Config.num_workers, help="Number of DataLoader workers.")
    parser.add_argument("--batch-size", type=int, default=Config.batch_size, help="Evaluation batch size.")
    parser.add_argument("--test-augment", dest="test_augment", action="store_true", help="Enable test-time augmentation.")
    parser.add_argument("--no-test-augment", dest="test_augment", action="store_false", help="Disable test-time augmentation.")
    parser.add_argument(
        "--use-latest-checkpoint",
        action="store_true",
        help="Use latest_checkpoint.pth when checkpoint-file and checkpoint-path are not provided.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device used for evaluation.")
    parser.set_defaults(test_augment=False)

    parser.add_argument("--data-root", default=DEFAULT_UNIFORM_DATA_ROOT, help="Uniform-grid dataset root directory.")
    parser.add_argument("--frame-dropout-rate", type=float, default=Config.frame_dropout_rate, help="Frame dropout rate used by the dataset pipeline.")
    parser.add_argument("--convlstm-hidden", type=int, default=Config.convlstm_hidden, help="ConvLSTM hidden channels.")
    parser.add_argument("--convlstm-layers", type=int, default=Config.convlstm_layers, help="Number of ConvLSTM layers.")
    parser.add_argument("--temporal-pool", choices=["mean", "max"], default=Config.temporal_pool, help="Temporal pooling strategy.")
    parser.add_argument("--temporal-stride", type=int, default=Config.temporal_stride, help="Temporal downsampling stride.")
    return parser


def make_summary_figure(true_hs, pred_hs, true_dir, pred_dir, metrics, save_path):
    diff_dir_signed = pred_dir - true_dir
    diff_dir_signed[diff_dir_signed > 180] -= 360
    diff_dir_signed[diff_dir_signed < -180] += 360
    diff_dir_abs = np.abs(diff_dir_signed)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=180)
    fig.suptitle("Uniform Grid Wave Height and Direction Evaluation Summary", fontsize=18, y=1.02)

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
        f"R2={metrics['r2_hs']:.3f}\nRMSE={metrics['rmse_hs']:.3f} m\nMAE={metrics['mae_hs']:.3f} m",
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
        f"R2={metrics['r2_dir']:.3f}\nRMSE={metrics['rmse_dir']:.3f} deg\nMAE={metrics['mae_dir']:.3f} deg",
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
        f"<=5 deg : {(diff_dir_abs <= 5).mean() * 100:.1f}%\n"
        f"<=10 deg: {(diff_dir_abs <= 10).mean() * 100:.1f}%\n"
        f"<=15 deg: {(diff_dir_abs <= 15).mean() * 100:.1f}%",
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
    logger, log_file = setup_logging(cfg.log_dir, "evaluate_convlstm_uniform")

    print(f"Device: {device}")
    print(f"Model name: {cfg.model_name}")
    print(f"Experiment name: {cfg.experiment_name}")
    print(f"Checkpoint path: {cfg.checkpoint_path}")
    print(f"Dataset root: {cfg.data_root}")
    print(f"Results directory: {cfg.results_dir}")
    print(f"Log file: {log_file}")
    print(f"Test augmentation: {cfg.test_augment}")

    logger.info(f"Evaluation started. Device: {device}")
    logger.info(f"Model name: {cfg.model_name}")
    logger.info(f"Experiment name: {cfg.experiment_name}")
    logger.info(f"Checkpoint path: {cfg.checkpoint_path}")
    logger.info(f"Dataset root: {cfg.data_root}")
    logger.info(f"Results directory: {cfg.results_dir}")
    logger.info(f"Test augmentation: {cfg.test_augment}")

    test_ds = LidarWaveDataset(mode="test", augment=cfg.test_augment, cfg=cfg)
    if len(test_ds) == 0:
        raise RuntimeError("Test dataset is empty. Please check --data-root and dataset preparation.")

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
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")

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

    checkpoint_stem = os.path.splitext(os.path.basename(cfg.checkpoint_path))[0]
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

    print(
        f"\nHs: R2={metrics['r2_hs']:.4f} | RMSE={metrics['rmse_hs']:.4f} m | "
        f"MAE={metrics['mae_hs']:.4f} m | Bias={metrics['bias_hs']:.4f} m"
    )
    print(
        f"Dir: R2={metrics['r2_dir']:.4f} | RMSE={metrics['rmse_dir']:.4f} deg | "
        f"MAE={metrics['mae_dir']:.4f} deg | Bias={metrics['bias_dir']:.4f} deg"
    )
    print(f"Detailed results: {csv_path}")
    print(f"Metrics table: {metrics_path}")
    print(f"Summary figure: {figure_path}")


def main():
    parser = create_parser()
    args = parser.parse_args()
    evaluate_and_plot(args)


if __name__ == "__main__":
    main()
