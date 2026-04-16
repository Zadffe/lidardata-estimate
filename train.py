import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from data.dataset import LidarWaveDataset
from models.CNN_ConvbLSTM import build_model


def resolve_resume_checkpoint_path(cfg, latest_checkpoint_path):
    resume_path = str(getattr(cfg, "resume_checkpoint_path", "") or "").strip()
    if resume_path:
        return resume_path

    if getattr(cfg, "resume_training", False):
        return latest_checkpoint_path

    return ""


def load_training_checkpoint(model, optimizer, scaler, checkpoint_path, device, logger):
    """
    加载完整训练状态。

    支持两种情况:
    1. 完整 checkpoint:
       包含模型、优化器、AMP、epoch、best_val_loss、loss 历史
    2. 纯模型权重:
       只恢复 model.state_dict()，从 epoch 0 重新计数继续训练
    """

    start_epoch = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    if not checkpoint_path:
        return start_epoch, best_val_loss, train_losses, val_losses, False

    if not os.path.exists(checkpoint_path):
        logger.warning(f"Resume checkpoint not found: {checkpoint_path}")
        return start_epoch, best_val_loss, train_losses, val_losses, False

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state:
            scaler.load_state_dict(scaler_state)

        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        train_losses = list(checkpoint.get("train_losses", []))
        val_losses = list(checkpoint.get("val_losses", []))

        logger.info(f"Resumed full checkpoint from: {checkpoint_path}")
        logger.info(
            f"Resume state: next_epoch={start_epoch + 1}, "
            f"best_val_loss={best_val_loss:.4f}, "
            f"history_len={len(train_losses)}"
        )
        return start_epoch, best_val_loss, train_losses, val_losses, True

    model.load_state_dict(checkpoint)
    logger.warning(
        "Loaded a model-only checkpoint for resume. "
        "Optimizer, scaler, epoch, and best_val_loss were not restored."
    )
    logger.warning(f"Model-only checkpoint path: {checkpoint_path}")
    return start_epoch, best_val_loss, train_losses, val_losses, True


def save_training_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scaler,
    epoch,
    best_val_loss,
    train_losses,
    val_losses,
    cfg,
):
    scaler_state = scaler.state_dict() if scaler.is_enabled() else None

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler_state,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "model_name": cfg.model_name,
        "experiment_tag": cfg.experiment_tag,
    }
    torch.save(checkpoint, checkpoint_path)


def train():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(cfg.log_dir, f"train_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    best_model_path = os.path.join(cfg.save_dir, "best_model.pth")
    latest_checkpoint_path = os.path.join(cfg.save_dir, "latest_checkpoint.pth")
    resume_checkpoint_path = resolve_resume_checkpoint_path(cfg, latest_checkpoint_path)

    print(f"训练设备: {device}")
    print(f"当前模型: {cfg.model_name}")
    print(f"权重保存目录: {cfg.save_dir}")
    print(f"最佳权重文件: {best_model_path}")
    print(f"续训检查点文件: {latest_checkpoint_path}")
    print(f"结果输出目录: {cfg.results_dir}")
    print(f"日志目录: {cfg.log_dir}")
    if resume_checkpoint_path:
        print(f"本次将尝试从该检查点恢复: {resume_checkpoint_path}")

    logger.info(f"Training started. Device: {device}")
    logger.info(f"Model name: {cfg.model_name}")
    logger.info(f"Checkpoint directory: {cfg.save_dir}")
    logger.info(f"Best checkpoint path: {best_model_path}")
    logger.info(f"Latest checkpoint path: {latest_checkpoint_path}")
    logger.info(f"Results directory: {cfg.results_dir}")
    logger.info(f"Log directory: {cfg.log_dir}")
    logger.info(
        "Config: "
        f"Epochs={cfg.epochs}, Batch={cfg.batch_size}, LR={cfg.lr}, "
        f"WeightDecay={cfg.weight_decay}, TemporalPool={cfg.temporal_pool}, "
        f"TemporalStride={cfg.temporal_stride}, Resume={cfg.resume_training}"
    )

    total_start_time = time.time()

    print("正在加载训练/验证数据集...")
    train_ds = LidarWaveDataset(mode="train", augment=True)
    val_ds = LidarWaveDataset(mode="val", augment=False)

    if len(train_ds) == 0:
        print("错误: 训练集为空，请检查 data_root 或数据生成流程。")
        return

    if len(val_ds) == 0:
        print("错误: 验证集为空，请检查 data_root 或数据生成流程。")
        return

    print(f"训练集样本数: {len(train_ds)} | 验证集样本数: {len(val_ds)}")
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
        batch_size=cfg.batch_size * 2,
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
        print(f"已恢复训练状态，将从 epoch {start_epoch + 1} 开始继续训练。")
    else:
        print("未恢复历史训练状态，将从头开始训练。")

    if start_epoch >= cfg.epochs:
        logger.info(
            f"Checkpoint already reached target epochs. start_epoch={start_epoch}, cfg.epochs={cfg.epochs}"
        )
        print("检查点对应的训练轮次已经达到或超过当前 epochs 设置，本次无需继续训练。")
        return

    print(f"开始训练，共 {cfg.epochs} 个 epoch...")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        train_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} [Train]")

        for inputs, target_dir, target_hs in pbar:
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
            pbar.set_postfix(
                {
                    "T_Loss": f"{total_loss.item():.4f}",
                    "Dir": f"{loss_dir.item():.4f}",
                    "Hs": f"{loss_hs.item():.4f}",
                }
            )

        avg_train_loss = train_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0.0

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

        avg_val_loss = val_loss_sum / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        logger.info(
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model saved to: {best_model_path} | Val Loss: {best_val_loss:.4f}")

        if getattr(cfg, "save_latest_checkpoint", True):
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

        if (epoch + 1) % max(1, int(getattr(cfg, "checkpoint_every", 20))) == 0:
            epoch_ckpt = os.path.join(cfg.save_dir, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), epoch_ckpt)
            logger.info(f"Periodic checkpoint saved to: {epoch_ckpt}")

    print("正在保存 loss 曲线...")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)

    loss_save_path = os.path.join(cfg.results_dir, "loss_curve.png")
    plt.savefig(loss_save_path)
    plt.close()
    logger.info(f"Loss curve saved to: {loss_save_path}")

    logger.info("Training completed.")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)

    time_str = f"Total Training Time: {hours}h {minutes}m {seconds}s ({total_duration:.2f} seconds)"
    logger.info(f"=== {time_str} ===")
    print(time_str)


if __name__ == "__main__":
    train()
