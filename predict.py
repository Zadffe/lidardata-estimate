# predict.py
import torch
import numpy as np
import math
import os
from models.wave_cnn import build_model
from configs.config import Config
from data.dataset import LidarWaveDataset

def predict_sample():
    # 1. 加载配置
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 加载模型
    model = build_model(cfg).to(device)
    # 假设加载第 50 轮的模型 (请根据实际文件名修改)
    model_path = os.path.join(cfg.save_dir, f"best_model.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型: {model_path}")
    else:
        print("未找到模型文件，请先运行 train.py")
        return

    model.eval() # 切换到评估模式
    
    # 3. 拿一个样本来测试 (这里直接借用 Dataset 的逻辑读取文件)
    # 用 'test' 里的数据演示，并开启 augment=True 以模拟真实环境中的数据丢失
    ds = LidarWaveDataset(mode='test', augment=True) 
    
    # 随机取第 0 个样本
    input_tensor, true_dir, true_hs_norm = ds[10]
    
    # 增加 Batch 维度: [1, T, H, W] -> [1, 1, T, H, W]
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # 4. 推理
    with torch.no_grad():
        pred_dir, pred_hs_norm = model(input_batch)
    
    # 5. 数据反归一化与解析
    # A. 解析方向
    sin_val, cos_val = pred_dir[0].cpu().numpy()
    pred_angle = math.degrees(math.atan2(sin_val, cos_val))
    if pred_angle < 0: pred_angle += 360
    
    true_sin, true_cos = true_dir.numpy()
    true_angle = math.degrees(math.atan2(true_sin, true_cos))
    if true_angle < 0: true_angle += 360
    
    # B. 解析波高 (反归一化)
    pred_hs = pred_hs_norm.item() * cfg.max_hs
    true_hs = true_hs_norm.item() * cfg.max_hs
    
    # 6. 打印对比结果
    print("-" * 30)
    print(f"真实波高 Hs: {true_hs:.2f} m")
    print(f"预测波高 Hs: {pred_hs:.2f} m")
    print(f"误差: {abs(pred_hs - true_hs):.2f} m")
    print("-" * 30)
    print(f"真实方向: {true_angle:.1f}°")
    print(f"预测方向: {pred_angle:.1f}°")
    print(f"误差: {abs(pred_angle - true_angle):.1f}°")
    print("-" * 30)

if __name__ == '__main__':
    predict_sample()