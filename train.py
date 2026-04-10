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
from models.wave_cnn import build_model


def train():
    # 1. 读取配置与设备信息
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

    best_model_path = os.path.join(cfg.save_dir, "best_model.pth")

    print(f"训练设备: {device}")
    print(f"当前模型: {cfg.model_name}")
    print(f"权重保存目录: {cfg.save_dir}")
    print(f"最佳权重文件: {best_model_path}")
    print(f"结果输出目录: {cfg.results_dir}")
    print(f"日志目录: {cfg.log_dir}")

    logging.info(f"Training started. Device: {device}")
    logging.info(f"Model name: {cfg.model_name}")
    logging.info(f"Checkpoint directory: {cfg.save_dir}")
    logging.info(f"Best checkpoint path: {best_model_path}")
    logging.info(f"Results directory: {cfg.results_dir}")
    logging.info(f"Log directory: {cfg.log_dir}")
    logging.info(
        "Config: "
        f"Epochs={cfg.epochs}, Batch={cfg.batch_size}, LR={cfg.lr}, "
        f"WeightDecay={cfg.weight_decay}, TemporalPool={cfg.temporal_pool}, "
        f"TemporalStride={cfg.temporal_stride}"
    )

    total_start_time = time.time()

    # 2. 构建数据集
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
    logging.info(f"Dataset size - train: {len(train_ds)}, val: {len(val_ds)}")

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

    # 3. 构建模型
    model = build_model(cfg).to(device)
    logging.info(f"Model class: {model.__class__.__name__}")

    # 4. 损失函数、优化器、AMP
    criterion_dir = nn.MSELoss()
    criterion_hs = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    print(f"开始训练，共 {cfg.epochs} 个 epoch...")

    for epoch in range(cfg.epochs):
        # ==========================
        #      Training Phase
        # ==========================
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

                # 当前训练目标采用两个任务等权求和。
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

        # ==========================
        #     Validation Phase
        # ==========================
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

        logging.info(
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best model saved to: {best_model_path} | Val Loss: {best_val_loss:.4f}")

        if (epoch + 1) % 20 == 0:
            epoch_ckpt = os.path.join(cfg.save_dir, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), epoch_ckpt)
            logging.info(f"Periodic checkpoint saved to: {epoch_ckpt}")

    # 6. 保存 loss 曲线
    print("正在保存 loss 曲线...")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, cfg.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, cfg.epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)

    loss_save_path = os.path.join(cfg.results_dir, "loss_curve.png")
    plt.savefig(loss_save_path)
    plt.close()
    logging.info(f"Loss curve saved to: {loss_save_path}")

    logging.info("Training completed.")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)

    time_str = f"Total Training Time: {hours}h {minutes}m {seconds}s ({total_duration:.2f} seconds)"
    logging.info(f"=== {time_str} ===")
    print(time_str)


if __name__ == "__main__":
    train()
