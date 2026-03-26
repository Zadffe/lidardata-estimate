import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import os
import sys
from tqdm import tqdm
import logging
import time

# 导入你的模块
from configs.config import Config
from data.dataset import LidarWaveDataset
from models.wave_cnn import build_model

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

def circular_r2_score(y_true, y_pred):
    """
    计算循环变量（如角度 0-360）的 R2 分数
    传统的 R2 会认为 0 和 360 差异巨大，而实际上它们是一样的
    """
    # 1. 计算残差平方和 SS_res
    # 计算预测值与真实值之间的最小角度差
    diff = np.abs(y_true - y_pred)
    diff = np.minimum(diff, 360 - diff)
    ss_res = np.sum(diff ** 2)

    # 2. 计算总平方和 SS_tot
    # 首先需要计算真实值的"循环均值" (Circular Mean)
    y_true_rad = np.deg2rad(y_true)
    sin_sum = np.sum(np.sin(y_true_rad))
    cos_sum = np.sum(np.cos(y_true_rad))
    mean_angle = np.degrees(np.arctan2(sin_sum, cos_sum))
    if mean_angle < 0:
        mean_angle += 360
    
    # 计算真实值与均值之间的最小角度差
    diff_mean = np.abs(y_true - mean_angle)
    diff_mean = np.minimum(diff_mean, 360 - diff_mean)
    ss_tot = np.sum(diff_mean ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - ss_res / ss_tot

def evaluate_and_plot():
    # 1. 初始化
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 开始可视化评估 ===")
    
    # --- 配置日志 ---
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
        
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(cfg.log_dir, f'evaluate_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Evaluation started using model: {os.path.join(cfg.save_dir, 'best_model.pth')}")

    # 利用测试集数据进行评估
    # 开启 augment=True 以评估模型在模拟真实雷达噪声下的性能
    test_ds = LidarWaveDataset(mode='test', augment=True)
    if len(test_ds) == 0:
        print("错误: 测试集为空！")
        return
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers)

    # 3. 加载模型
    model = build_model(cfg).to(device)
    model_path = os.path.join(cfg.save_dir, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"未找到模型: {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. 数据收集容器
    true_hs_list, pred_hs_list = [], []
    true_dir_list, pred_dir_list = [], []

    # 5. 推理循环
    with torch.no_grad():
        for inputs, target_dir, target_hs in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            
            # 模型预测
            pred_dir_vec, pred_hs_norm = model(inputs)
            
            # --- 解析波高 ---
            pred_hs = pred_hs_norm.cpu().numpy().flatten() * cfg.max_hs
            true_hs = target_hs.numpy().flatten() * cfg.max_hs
            
            true_hs_list.extend(true_hs)
            pred_hs_list.extend(pred_hs)

            # --- 解析方向 (角度) ---
            # 预测角度
            p_sin = pred_dir_vec[:, 0].cpu().numpy()
            p_cos = pred_dir_vec[:, 1].cpu().numpy()
            p_ang = np.degrees(np.arctan2(p_sin, p_cos))
            p_ang[p_ang < 0] += 360
            
            # 真实角度
            t_sin = target_dir[:, 0].numpy()
            t_cos = target_dir[:, 1].numpy()
            t_ang = np.degrees(np.arctan2(t_sin, t_cos))
            t_ang[t_ang < 0] += 360
            
            true_dir_list.extend(t_ang)
            pred_dir_list.extend(p_ang)

    # 转换为 Numpy 数组
    true_hs = np.array(true_hs_list)
    pred_hs = np.array(pred_hs_list)
    true_dir = np.array(true_dir_list)
    pred_dir = np.array(pred_dir_list)

    # --- 保存预测结果到 CSV ---
    if not os.path.exists(cfg.results_dir):
        os.makedirs(cfg.results_dir)
        
    csv_timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(cfg.results_dir, f'evaluation_results_{csv_timestamp}.csv')
    
    df_results = pd.DataFrame({
        'True_Hs': true_hs,
        'Pred_Hs': pred_hs,
        'True_Dir': true_dir,
        'Pred_Dir': pred_dir,
        'Hs_Error': pred_hs - true_hs,
        'Dir_Error_Abs': np.minimum(np.abs(true_dir - pred_dir), 360 - np.abs(true_dir - pred_dir))
    })
    
    df_results.to_csv(csv_path, index=False)
    logging.info(f"Detailed prediction results saved to: {csv_path}")
    print(f"预测结果明细已保存至: {csv_path}")

    # 6. 计算 R2 分数 (模型拟合度)
    # --- Wave Height (Hs) 指标计算 ---
    r2_hs = r2_score(true_hs, pred_hs)
    rmse_hs = np.sqrt(mean_squared_error(true_hs, pred_hs))
    mae_hs = mean_absolute_error(true_hs, pred_hs)
    bias_hs = np.mean(pred_hs - true_hs) # 偏差 = 预测 - 真实
    
    # --- Wave Direction (Dir) 指标计算 ---
    r2_dir = circular_r2_score(true_dir, pred_dir)
    
    # 计算角度差 (0-180)，用于 RMSE 和 MAE
    diff_dir_abs = np.abs(true_dir - pred_dir)
    diff_dir_abs = np.minimum(diff_dir_abs, 360 - diff_dir_abs)
    
    rmse_dir = np.sqrt(np.mean(diff_dir_abs**2))
    mae_dir = np.mean(diff_dir_abs)

    # 计算有符号的角度偏差 (用于 Bias)
    diff_dir_signed = pred_dir - true_dir
    diff_dir_signed[diff_dir_signed > 180] -= 360
    diff_dir_signed[diff_dir_signed < -180] += 360
    bias_dir = np.mean(diff_dir_signed)
    
    # 记录评估指标
    logging.info(f"=== Detailed Evaluation Metrics ===")
    logging.info(f"[Wave Height Hs]")
    logging.info(f"  R2 Score: {r2_hs:.4f}")
    logging.info(f"  RMSE:     {rmse_hs:.4f} m")
    logging.info(f"  MAE:      {mae_hs:.4f} m")
    
    logging.info(f"[Wave Direction Dir]")
    logging.info(f"  Circ R2:  {r2_dir:.4f}")
    logging.info(f"  RMSE:     {rmse_dir:.4f} deg")
    logging.info(f"  MAE:      {mae_dir:.4f} deg")
    
    print(f"\n波高 R2: {r2_hs:.4f} | RMSE: {rmse_hs:.4f} | MAE: {mae_hs:.4f}")
    print(f"波向 R2: {r2_dir:.4f} | RMSE: {rmse_dir:.4f} | MAE: {mae_dir:.4f}")

    # ==========================================
    #             开始绘图 (关键部分)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # --- 图 1: 波高散点图 (Regression Scatter Plot) ---
    ax = axes[0, 0]
    sns.scatterplot(x=true_hs, y=pred_hs, ax=ax, alpha=0.6, color='blue', edgecolor='k')
    # 画对角线 y=x
    lims = [0, 6] # 你的波高范围
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction Line')
    ax.set_title(f'Wave height prediction analysis ($R^2$={r2_hs:.3f})', fontsize=14)
    ax.set_xlabel('Real Height (m)', fontsize=12)
    ax.set_ylabel('Predicted Height (m)', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # --- 图 2: 波高误差分布 (Residual Histogram) ---
    ax = axes[0, 1]
    error_hs = pred_hs - true_hs
    sns.histplot(error_hs, kde=True, ax=ax, color='green', bins=30)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_title('Height Error Distribution', fontsize=14)
    ax.set_xlabel('Prediction Error (m)', fontsize=12)
    ax.set_ylabel('Sample size', fontsize=12)
    
    # --- 图 3: 波向散点图 ---
    ax = axes[1, 0]
    sns.scatterplot(x=true_dir, y=pred_dir, ax=ax, alpha=0.6, color='purple', edgecolor='k')
    ax.plot([0, 360], [0, 360], 'r--', linewidth=2)
    ax.set_title(f'Wave direction prediction analysis ($R^2$={r2_dir:.3f})', fontsize=14)
    ax.set_xlabel('Real angle (°)', fontsize=12)
    ax.set_ylabel('Prediction angle (°)', fontsize=12)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.grid(True)
    
    # 波向极坐标误差图 (Polar Error) ---
    # 这是一个比较高级的图，展示方向预测的偏差
    # 如果点都在圆心附近，说明误差极小
    ax = axes[1, 1]
    # 计算角度差 (处理 0/360 跳变)
    diff = pred_dir - true_dir
    diff[diff > 180] -= 360
    diff[diff < -180] += 360
    
    sns.histplot(diff, kde=True, ax=ax, color='orange', bins=30)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_title('Angular error distribution', fontsize=14)
    ax.set_xlabel('Angular error (°)', fontsize=12)
    
    plt.tight_layout()
    if not os.path.exists(cfg.results_dir):
        os.makedirs(cfg.results_dir)
    save_path = os.path.join(cfg.results_dir, "scatter_plot.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n可视化报告已保存至: {save_path}")
    print("请打开图片查看模型性能！")
    plt.show()

if __name__ == '__main__':
    evaluate_and_plot()