# train.py (包含验证集评估与最佳模型保存功能)
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import time

from configs.config import Config
from data.dataset import LidarWaveDataset
from models.wave_cnn import build_model

def train():
    # 1. 初始化配置
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
        
    # --- 配置日志 ---
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(cfg.log_dir, f'train_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Training started. Device: {device}")
    logging.info(f"Config: Epochs={cfg.epochs}, Batch={cfg.batch_size}, LR={cfg.lr}")
    
    # 记录开始时间
    total_start_time = time.time()

    # 2. 准备数据 (训练集 + 验证集)
    print("正在加载数据集...")
    # 开启 augment=True 以模拟真实雷达的动态数据丢失
    train_ds = LidarWaveDataset(mode='train', augment=False)
    # 验证集通常保持纯净，或者也可以开启 augment 来评估鲁棒性
    val_ds = LidarWaveDataset(mode='val', augment=False)
    
    # 检查验证集是否为空
    if len(val_ds) == 0:
        print("错误: 验证集为空！请先运行 Data Generator 生成验证集 (设置 n_val > 0)。")
        return

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(cfg.num_workers > 0)
    )
    # 验证集不需要 shuffle，且 batch_size 可以大一点因为不占反向传播显存
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(cfg.num_workers > 0)
    )
    
    # 3. 初始化模型
    model = build_model(cfg).to(device)
    
    # 4. 定义损失函数和优化器
    criterion_dir = nn.MSELoss() 
    criterion_hs = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # 记录最佳 Loss，用于保存最佳模型
    best_val_loss = float('inf')

    # 用于记录每个epoch的loss，以便绘制曲线
    train_losses = []
    val_losses = []

    # 5. 开始训练循环
    print(f"开始训练，共 {cfg.epochs} 轮...")
    
    for epoch in range(cfg.epochs):
        # ==========================
        #      Training Phase
        # ==========================
        model.train() # 开启训练模式 (启用 Dropout)
        train_loss_sum = 0.0
        
        # 训练进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]")
        
        for inputs, target_dir, target_hs in pbar:
            inputs = inputs.to(device, non_blocking=True)
            target_dir = target_dir.to(device, non_blocking=True)
            target_hs = target_hs.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                # 前向传播
                pred_dir, pred_hs = model(inputs)

                # 计算 Loss
                loss_dir = criterion_dir(pred_dir, target_dir)
                loss_hs = criterion_hs(pred_hs, target_hs)

                # 这里的 0.5 是权重，可以根据需要调整。
                # 比如你觉得方向更重要，可以: loss_dir + 0.2 * loss_hs
                total_loss = loss_dir + 0.3 * loss_hs

            # 反向传播
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sum += total_loss.item()
            pbar.set_postfix({'T_Loss': total_loss.item()})
            
        avg_train_loss = train_loss_sum / len(train_loader)

        # ==========================
        #     Validation Phase
        # ==========================
        model.eval() # 开启评估模式 (关闭 Dropout, 锁定 BN)
        val_loss_sum = 0.0
        
        # 验证不需要计算梯度，节省显存和时间
        with torch.no_grad():
            for inputs, target_dir, target_hs in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                target_dir = target_dir.to(device, non_blocking=True)
                target_hs = target_hs.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    pred_dir, pred_hs = model(inputs)

                    loss_dir = criterion_dir(pred_dir, target_dir)
                    loss_hs = criterion_hs(pred_hs, target_hs)

                    val_loss = loss_dir + loss_hs
                val_loss_sum += val_loss.item()
        
        avg_val_loss = val_loss_sum / len(val_loader)
        
        # 记录 loss
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 打印本轮总结
        log_msg = f"Epoch {epoch+1}/{cfg.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        logging.info(log_msg)
        
        # Save Best Model
        # 如果当前验证集 Loss 比历史最低还低，说明模型进步了，保存它
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(cfg.save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            logging.info(f"★ Best Model Saved! Val Loss: {best_val_loss:.4f}")
        
        # 另外，每 10 轮保存一个常规存档，防止意外中断
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, f"epoch_{epoch+1}.pth"))

    # 绘制并保存 Loss 曲线
    print("正在绘制并保存损失曲线...")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, cfg.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, cfg.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # 确保保存路径存在
    if not os.path.exists(cfg.results_dir):
        os.makedirs(cfg.results_dir)
        
    loss_save_path = os.path.join(cfg.results_dir, 'loss_curve.png')
    plt.savefig(loss_save_path)
    plt.close()
    logging.info(f"Loss curve saved to: {loss_save_path}")

    logging.info("Training completed.")
    
    # 计算并记录总耗时
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    time_str = f"Total Training Time: {hours}h {minutes}m {seconds}s ({total_duration:.2f} seconds)"
    logging.info(f"=== {time_str} ===")
    print(time_str)

if __name__ == '__main__':
    train()