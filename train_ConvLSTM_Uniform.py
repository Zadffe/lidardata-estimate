import argparse
import logging
import os
import random
import sys
import time
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from data.dataset import LidarWaveDataset
from models.CNN_ConvLSTM_Uniform import build_model
from utils.checkpoint_utils import (
    load_training_checkpoint,
    resolve_resume_checkpoint_path,
    save_training_checkpoint,
)


DEFAULT_UNIFORM_DATA_ROOT = r"F:\Research__dir\dl_lidar\datasets\Dataset_Wave_Lidar_10000samples_uniform_grid"


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


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_swanlab(cfg):
    if not getattr(cfg, "enable_swanlab", False):
        return None

    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError(
            "swanlab is not installed. Please run `pip install swanlab` first, or disable --enable-swanlab."
        ) from exc

    experiment_name = str(getattr(cfg, "swanlab_experiment_name", "") or "").strip() or cfg.experiment_name
    tags_raw = str(getattr(cfg, "swanlab_tags", "") or "").strip()
    tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]

    config_dict = {
        "model_name": cfg.model_name,
        "experiment_name": cfg.experiment_name,
        "data_root": cfg.data_root,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "num_workers": cfg.num_workers,
        "seed": cfg.seed,
        "device": str(cfg.device),
        "frames": cfg.frames,
        "height": cfg.height,
        "width": cfg.width,
        "lidar_scale": cfg.lidar_scale,
        "max_hs": cfg.max_hs,
        "frame_dropout_rate": cfg.frame_dropout_rate,
        "temporal_pool": cfg.temporal_pool,
        "temporal_stride": cfg.temporal_stride,
        "convlstm_hidden": cfg.convlstm_hidden,
        "convlstm_layers": cfg.convlstm_layers,
        "train_augment": cfg.train_augment,
    }

    run = swanlab.init(
        project=cfg.swanlab_project,
        workspace=(str(cfg.swanlab_workspace).strip() or None),
        experiment_name=experiment_name,
        tags=tags or None,
        config=config_dict,
        logdir=cfg.log_dir,
    )
    return run


def create_parser():
    parser = argparse.ArgumentParser(
        description="Train the ConvLSTM model for uniform-grid wave data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output-root", type=str, default=Config.output_root, help="Root directory for experiment outputs.")
    parser.add_argument("--experiment-name", type=str, default=get_default_experiment_name(), help="Experiment directory name.")
    parser.add_argument("--epochs", type=int, default=Config.epochs, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=Config.batch_size, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=Config.lr, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay, help="AdamW weight decay.")
    parser.add_argument("--num-workers", type=int, default=Config.num_workers, help="Number of DataLoader workers.")
    parser.add_argument("--save-interval", type=int, default=Config.save_interval, help="Save a periodic checkpoint every N epochs.")
    parser.add_argument("--seed", type=int, default=Config.seed, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device.")

    parser.add_argument("--data-root", type=str, default=DEFAULT_UNIFORM_DATA_ROOT, help="Uniform-grid dataset root directory.")
    parser.add_argument("--frame-dropout-rate", type=float, default=Config.frame_dropout_rate, help="Frame dropout rate used if training augmentation is enabled.")

    parser.add_argument("--temporal-pool", type=str, default=Config.temporal_pool, choices=["mean", "max"], help="Temporal pooling mode.")
    parser.add_argument("--temporal-stride", type=int, default=Config.temporal_stride, help="Temporal downsampling stride.")
    parser.add_argument("--convlstm-hidden", type=int, default=Config.convlstm_hidden, help="ConvLSTM hidden channels.")
    parser.add_argument("--convlstm-layers", type=int, default=Config.convlstm_layers, help="Number of ConvLSTM layers.")

    parser.add_argument("--resume-training", action="store_true", help="Resume from the default latest checkpoint.")
    parser.add_argument("--resume-checkpoint-path", type=str, default="", help="Resume from an explicit checkpoint path.")
    parser.add_argument("--save-latest-checkpoint", dest="save_latest_checkpoint", action="store_true", help="Save latest_checkpoint.pth during training.")
    parser.add_argument("--no-save-latest-checkpoint", dest="save_latest_checkpoint", action="store_false", help="Disable latest checkpoint saving.")
    parser.add_argument("--enable-train-augment", dest="train_augment", action="store_true", help="Enable the existing dataset augmentation pipeline.")
    parser.add_argument("--disable-train-augment", dest="train_augment", action="store_false", help="Disable training-time augmentation.")
    parser.add_argument("--enable-swanlab", action="store_true", help="Log training metrics to SwanLab.")
    parser.add_argument("--swanlab-project", type=str, default="dl-lidar-uniform-grid", help="SwanLab project name.")
    parser.add_argument("--swanlab-workspace", type=str, default="", help="Optional SwanLab workspace or organization username.")
    parser.add_argument("--swanlab-experiment-name", type=str, default="", help="Optional SwanLab experiment name. Defaults to experiment-name.")
    parser.add_argument("--swanlab-tags", type=str, default="", help="Optional comma-separated SwanLab tags.")
    parser.set_defaults(save_latest_checkpoint=Config.save_latest_checkpoint, train_augment=False)

    return parser


def build_cfg_from_args(args):
    base_cfg = Config()
    experiment_name = str(args.experiment_name or "").strip() or get_default_experiment_name()
    _, save_dir, results_dir, log_dir = build_experiment_dirs(args.output_root, experiment_name)

    return SimpleNamespace(
        model_name="ConvLSTMUniformGrid",
        output_root=args.output_root,
        experiment_name=experiment_name,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        checkpoint_every=args.save_interval,
        seed=args.seed,
        device=args.device,
        frames=base_cfg.frames,
        height=base_cfg.height,
        width=base_cfg.width,
        lidar_scale=base_cfg.lidar_scale,
        max_hs=base_cfg.max_hs,
        frame_dropout_rate=args.frame_dropout_rate,
        temporal_pool=args.temporal_pool,
        temporal_stride=args.temporal_stride,
        convlstm_hidden=args.convlstm_hidden,
        convlstm_layers=args.convlstm_layers,
        resume_training=args.resume_training,
        resume_checkpoint_path=args.resume_checkpoint_path,
        save_latest_checkpoint=args.save_latest_checkpoint,
        train_augment=args.train_augment,
        enable_swanlab=args.enable_swanlab,
        swanlab_project=args.swanlab_project,
        swanlab_workspace=args.swanlab_workspace,
        swanlab_experiment_name=args.swanlab_experiment_name,
        swanlab_tags=args.swanlab_tags,
        save_dir=save_dir,
        results_dir=results_dir,
        log_dir=log_dir,
    )


def run_training(cfg):
    set_random_seed(cfg.seed)
    device = resolve_device(cfg.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    logger, log_file = setup_logging(cfg.log_dir, "train_convlstm_uniform")
    swanlab_run = init_swanlab(cfg)
    best_model_path = os.path.join(cfg.save_dir, "best_model.pth")
    latest_checkpoint_path = os.path.join(cfg.save_dir, "latest_checkpoint.pth")
    resume_checkpoint_path = resolve_resume_checkpoint_path(cfg, latest_checkpoint_path)

    print(f"Training device: {device}")
    print(f"Model name: {cfg.model_name}")
    print(f"Experiment name: {cfg.experiment_name}")
    print(f"Dataset root: {cfg.data_root}")
    print(f"Checkpoint directory: {cfg.save_dir}")
    print(f"Best model path: {best_model_path}")
    print(f"Latest checkpoint path: {latest_checkpoint_path}")
    print(f"Results directory: {cfg.results_dir}")
    print(f"Log file: {log_file}")
    print(f"Train augmentation: {cfg.train_augment}")
    print(f"SwanLab enabled: {cfg.enable_swanlab}")
    print(f"Frame dropout rate: {cfg.frame_dropout_rate:.2f}")
    if resume_checkpoint_path:
        print(f"Resume checkpoint: {resume_checkpoint_path}")

    logger.info(
        f"Config: epochs={cfg.epochs}, batch_size={cfg.batch_size}, lr={cfg.lr}, "
        f"weight_decay={cfg.weight_decay}, train_augment={cfg.train_augment}, "
        f"frame_dropout_rate={cfg.frame_dropout_rate}, temporal_pool={cfg.temporal_pool}, "
        f"temporal_stride={cfg.temporal_stride}, convlstm_hidden={cfg.convlstm_hidden}, "
        f"convlstm_layers={cfg.convlstm_layers}, seed={cfg.seed}, data_root={cfg.data_root}"
    )

    train_ds = LidarWaveDataset(mode="train", augment=cfg.train_augment, cfg=cfg)
    val_ds = LidarWaveDataset(mode="val", augment=False, cfg=cfg)

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty. Please check --data-root and the train directory.")
    if len(val_ds) == 0:
        raise RuntimeError("Validation dataset is empty. Please check --data-root and the val directory.")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    model = build_model(cfg).to(device)
    criterion_dir = nn.MSELoss()
    criterion_hs = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch, best_val_loss, train_losses, val_losses, resumed = load_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        checkpoint_path=resume_checkpoint_path,
        device=device,
        logger=logger,
    )

    if resumed:
        print(f"Resumed training from epoch {start_epoch + 1}.")
    else:
        print("No resumable state found. Training will start from scratch.")

    if start_epoch >= cfg.epochs:
        print("Current checkpoint already reached or exceeded the configured epoch count. No further training will run.")
        return

    total_start_time = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        train_loss_sum = 0.0
        train_dir_loss_sum = 0.0
        train_hs_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} [{cfg.model_name}]", leave=False)
        for inputs, target_dir, target_hs in pbar:
            inputs = inputs.to(device, non_blocking=True)
            target_dir = target_dir.to(device, non_blocking=True)
            target_hs = target_hs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred_dir, pred_hs = model(inputs)
                loss_dir = criterion_dir(pred_dir, target_dir)
                loss_hs = criterion_hs(pred_hs, target_hs)
                total_loss = loss_dir + loss_hs

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += total_loss.item()
            train_dir_loss_sum += loss_dir.item()
            train_hs_loss_sum += loss_hs.item()

            pbar.set_postfix(
                {
                    "T_Loss": f"{total_loss.item():.4f}",
                    "Dir": f"{loss_dir.item():.4f}",
                    "Hs": f"{loss_hs.item():.4f}",
                }
            )

        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_dir_loss = train_dir_loss_sum / len(train_loader)
        avg_train_hs_loss = train_hs_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0.0
        val_dir_loss_sum = 0.0
        val_hs_loss_sum = 0.0

        with torch.no_grad():
            for inputs, target_dir, target_hs in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                target_dir = target_dir.to(device, non_blocking=True)
                target_hs = target_hs.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    pred_dir, pred_hs = model(inputs)
                    loss_dir = criterion_dir(pred_dir, target_dir)
                    loss_hs = criterion_hs(pred_hs, target_hs)
                    val_loss = loss_dir + loss_hs

                val_loss_sum += val_loss.item()
                val_dir_loss_sum += loss_dir.item()
                val_hs_loss_sum += loss_hs.item()

        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_dir_loss = val_dir_loss_sum / len(val_loader)
        avg_val_hs_loss = val_hs_loss_sum / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        summary = (
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"Train Loss={avg_train_loss:.4f} (Dir={avg_train_dir_loss:.4f}, Hs={avg_train_hs_loss:.4f}) | "
            f"Val Loss={avg_val_loss:.4f} (Dir={avg_val_dir_loss:.4f}, Hs={avg_val_hs_loss:.4f})"
        )
        logger.info(summary)
        if swanlab_run is not None:
            swanlab_run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": avg_train_loss,
                    "train/dir_loss": avg_train_dir_loss,
                    "train/hs_loss": avg_train_hs_loss,
                    "val/loss": avg_val_loss,
                    "val/dir_loss": avg_val_dir_loss,
                    "val/hs_loss": avg_val_hs_loss,
                    "best_val_loss": min(best_val_loss, avg_val_loss),
                },
                step=epoch + 1,
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model saved to: {best_model_path} | Val Loss={best_val_loss:.4f}")

        if cfg.save_latest_checkpoint:
            save_training_checkpoint(
                checkpoint_path=latest_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                train_losses=train_losses,
                val_losses=val_losses,
                cfg=cfg,
            )

        if (epoch + 1) % max(1, int(cfg.checkpoint_every)) == 0:
            epoch_ckpt = os.path.join(cfg.save_dir, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), epoch_ckpt)
            logger.info(f"Periodic checkpoint saved to: {epoch_ckpt}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("ConvLSTM Uniform Grid Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "loss_curve.png"), dpi=300)
    plt.close()

    total_duration = time.time() - total_start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    duration_msg = f"Total training time: {hours}h {minutes}m {seconds}s ({total_duration:.2f} seconds)"
    print(duration_msg)
    logger.info(duration_msg)
    if swanlab_run is not None:
        swanlab_run.log(
            {
                "final/total_training_time_seconds": total_duration,
                "final/best_val_loss": best_val_loss,
            },
            step=cfg.epochs,
        )
        swanlab_run.finish()


def main():
    parser = create_parser()
    args = parser.parse_args()
    cfg = build_cfg_from_args(args)
    run_training(cfg)


if __name__ == "__main__":
    main()
