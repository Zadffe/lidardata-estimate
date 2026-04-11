import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import LidarWaveDataset
from models.wave_cnn import build_model
from utils.checkpoint_utils import (
    load_training_checkpoint,
    resolve_resume_checkpoint_path,
    save_training_checkpoint,
)
from utils.runtime_utils import (
    build_train_cfg_from_args,
    resolve_device,
    set_random_seed,
    setup_logging,
)


def run_training(model_name, args):
    cfg = build_train_cfg_from_args(args, model_name)
    set_random_seed(cfg.seed)

    if cfg.model_name == "ConvLSTM" and cfg.temporal_pool == "cls":
        raise ValueError("ConvLSTM 模型不支持 temporal_pool='cls'，请使用 mean 或 max。")

    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    logger, log_file = setup_logging(cfg.log_dir, f"train_{cfg.model_name.lower()}")

    best_model_path = os.path.join(cfg.save_dir, "best_model.pth")
    latest_checkpoint_path = os.path.join(cfg.save_dir, "latest_checkpoint.pth")
    resume_checkpoint_path = resolve_resume_checkpoint_path(cfg, latest_checkpoint_path)

    print(f"训练设备: {device}")
    print(f"当前模型: {cfg.model_name}")
    print(f"实验名称: {cfg.experiment_name}")
    print(f"权重保存目录: {cfg.save_dir}")
    print(f"最佳模型路径: {best_model_path}")
    print(f"最新断点路径: {latest_checkpoint_path}")
    print(f"结果保存目录: {cfg.results_dir}")
    print(f"日志文件: {log_file}")
    if resume_checkpoint_path:
        print(f"续训权重来源: {resume_checkpoint_path}")

    logger.info(f"Training started. Device: {device}")
    logger.info(f"Model name: {cfg.model_name}")
    logger.info(f"Experiment name: {cfg.experiment_name}")
    logger.info(f"Checkpoint directory: {cfg.save_dir}")
    logger.info(f"Best checkpoint path: {best_model_path}")
    logger.info(f"Latest checkpoint path: {latest_checkpoint_path}")
    logger.info(f"Results directory: {cfg.results_dir}")
    logger.info(
        "Config: "
        f"Epochs={cfg.epochs}, Batch={cfg.batch_size}, LR={cfg.lr}, "
        f"WeightDecay={cfg.weight_decay}, TemporalPool={cfg.temporal_pool}, "
        f"TemporalStride={cfg.temporal_stride}, Seed={cfg.seed}"
    )

    total_start_time = time.time()

    train_ds = LidarWaveDataset(mode="train", augment=cfg.train_augment, cfg=cfg)
    val_ds = LidarWaveDataset(mode="val", augment=False, cfg=cfg)

    if len(train_ds) == 0:
        raise RuntimeError("训练集为空，请检查 --data-root 路径以及 train 目录内容。")
    if len(val_ds) == 0:
        raise RuntimeError("验证集为空，请检查 --data-root 路径以及 val 目录内容。")

    logger.info(f"Dataset size - train: {len(train_ds)}, val: {len(val_ds)}")

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
    logger.info(f"Model class: {model.__class__.__name__}")

    criterion_dir = nn.MSELoss()
    criterion_hs = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch, best_val_loss, train_losses, val_losses, resumed = load_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        checkpoint_path=resume_checkpoint_path,
        device=device,
        logger=logger,
    )

    if resumed:
        print(f"已恢复训练，将从 epoch {start_epoch + 1} 开始继续。")
    else:
        print("未检测到可恢复状态，本次将从头开始训练。")

    if start_epoch >= cfg.epochs:
        logger.info(
            f"Checkpoint already reached target epochs. start_epoch={start_epoch}, cfg.epochs={cfg.epochs}"
        )
        print("当前 checkpoint 已经达到或超过设定训练轮数，本次不再继续训练。")
        return

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        train_loss_sum = 0.0
        train_dir_loss_sum = 0.0
        train_hs_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} [{cfg.model_name}]")

        for batch_idx, (inputs, target_dir, target_hs) in enumerate(pbar, start=1):
            inputs = inputs.to(device, non_blocking=True)
            target_dir = target_dir.to(device, non_blocking=True)
            target_hs = target_hs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
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

            if batch_idx % cfg.log_interval == 0 or batch_idx == len(train_loader):
                logger.info(
                    f"Epoch {epoch + 1}/{cfg.epochs} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Train Loss={total_loss.item():.4f} | Dir={loss_dir.item():.4f} | Hs={loss_hs.item():.4f}"
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

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
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

        logger.info(
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} (Dir={avg_train_dir_loss:.4f}, Hs={avg_train_hs_loss:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} (Dir={avg_val_dir_loss:.4f}, Hs={avg_val_hs_loss:.4f})"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model saved to: {best_model_path} | Val Loss: {best_val_loss:.4f}")

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
    plt.title(f"{cfg.model_name} Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    loss_save_path = os.path.join(cfg.results_dir, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(loss_save_path, dpi=300)
    plt.close()
    logger.info(f"Loss curve saved to: {loss_save_path}")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    time_str = f"总训练时长: {hours}h {minutes}m {seconds}s ({total_duration:.2f} seconds)"
    logger.info(f"=== {time_str} ===")
    print(time_str)
