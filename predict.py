import math
import os

import torch

from configs.config import Config
from data.dataset import LidarWaveDataset
from models.CNN_ConvbLSTM import build_model


def resolve_inference_checkpoint_path(cfg):
    explicit_path = str(getattr(cfg, "inference_checkpoint_path", "") or "").strip()
    if explicit_path:
        return explicit_path

    if getattr(cfg, "inference_use_latest_checkpoint", False):
        return os.path.join(cfg.save_dir, "latest_checkpoint.pth")

    return os.path.join(cfg.save_dir, "best_model.pth")


def load_inference_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return "full_checkpoint"

    model.load_state_dict(checkpoint)
    return "model_state_dict"


def predict_sample():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_inference_checkpoint_path(cfg)

    print(f"推理设备: {device}")
    print(f"当前模型: {cfg.model_name}")
    print(f"权重文件: {checkpoint_path}")

    model = build_model(cfg).to(device)
    if not os.path.exists(checkpoint_path):
        print(f"错误: 未找到权重文件 {checkpoint_path}")
        return

    checkpoint_type = load_inference_weights(model, checkpoint_path, device)
    print(f"已加载权重，格式: {checkpoint_type}")
    model.eval()

    ds = LidarWaveDataset(mode="test", augment=cfg.test_augment)
    if len(ds) == 0:
        print("错误: 测试集为空。")
        return

    sample_index = min(10, len(ds) - 1)
    input_tensor, true_dir, true_hs_norm = ds[sample_index]
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_dir, pred_hs_norm = model(input_batch)

    sin_val, cos_val = pred_dir[0].cpu().numpy()
    pred_angle = math.degrees(math.atan2(sin_val, cos_val))
    if pred_angle < 0:
        pred_angle += 360

    true_sin, true_cos = true_dir.numpy()
    true_angle = math.degrees(math.atan2(true_sin, true_cos))
    if true_angle < 0:
        true_angle += 360

    pred_hs = pred_hs_norm.item() * cfg.max_hs
    true_hs = true_hs_norm.item() * cfg.max_hs

    print("-" * 30)
    print(f"样本索引: {sample_index}")
    print(f"真实波高 Hs: {true_hs:.2f} m")
    print(f"预测波高 Hs: {pred_hs:.2f} m")
    print(f"波高误差: {abs(pred_hs - true_hs):.2f} m")
    print("-" * 30)
    print(f"真实波向: {true_angle:.1f} deg")
    print(f"预测波向: {pred_angle:.1f} deg")
    print(f"波向误差: {abs(pred_angle - true_angle):.1f} deg")
    print("-" * 30)


if __name__ == "__main__":
    predict_sample()
