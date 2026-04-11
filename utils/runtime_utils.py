import logging
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from configs.config import Config


def config_to_dict(cfg):
    return {
        name: getattr(cfg, name)
        for name in dir(cfg)
        if not name.startswith("_") and not callable(getattr(cfg, name))
    }


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


def build_runtime_cfg(base_cfg, overrides):
    cfg_dict = config_to_dict(base_cfg)
    cfg_dict.update(overrides)
    return SimpleNamespace(**cfg_dict)


def get_default_experiment_name(model_name):
    normalized = Config._normalize_model_dir_name(model_name)
    mapping = {
        "ConvLSTM": "convlstm_default",
        "CNN": "cnn_default",
        "TemporalTransformer": "transformer_default",
    }
    return mapping.get(normalized, f"{normalized.lower()}_default")


def build_experiment_dirs(output_root, experiment_name):
    exp_root = os.path.join(output_root, experiment_name)
    save_dir = os.path.join(exp_root, "checkpoints")
    results_dir = os.path.join(exp_root, "results")
    log_dir = os.path.join(exp_root, "logs")
    return exp_root, save_dir, results_dir, log_dir


def add_common_data_args(parser):
    parser.add_argument(
        "--data-root",
        default=Config.data_root,
        help="数据集根目录，目录下应包含 train/val/test 子目录。",
    )
    parser.add_argument("--frames", type=int, default=Config.frames, help="每个样本包含的时间帧数。")
    parser.add_argument("--height", type=int, default=Config.height, help="输入张量高度。")
    parser.add_argument("--width", type=int, default=Config.width, help="输入张量宽度。")
    parser.add_argument("--lidar-scale", type=float, default=Config.lidar_scale, help="LiDAR 高度值归一化尺度。")
    parser.add_argument("--max-hs", type=float, default=Config.max_hs, help="波高 Hs 的反归一化上限。")


def add_convlstm_args(parser):
    parser.add_argument("--convlstm-hidden", type=int, default=Config.convlstm_hidden, help="ConvLSTM 隐藏通道数。")
    parser.add_argument("--convlstm-layers", type=int, default=Config.convlstm_layers, help="ConvLSTM 层数。")


def add_transformer_args(parser):
    parser.add_argument("--vit-embed-dim", type=int, default=Config.vit_embed_dim, help="Transformer token 维度。")
    parser.add_argument("--vit-depth", type=int, default=Config.vit_depth, help="Transformer block 数量。")
    parser.add_argument("--vit-num-heads", type=int, default=Config.vit_num_heads, help="多头注意力头数。")
    parser.add_argument("--vit-mlp-ratio", type=float, default=Config.vit_mlp_ratio, help="Transformer MLP 扩展倍率。")
    parser.add_argument("--vit-dropout", type=float, default=Config.vit_dropout, help="Transformer dropout。")
    parser.add_argument("--vit-attn-dropout", type=float, default=Config.vit_attn_dropout, help="注意力层 dropout。")


def add_common_eval_args(parser):
    default_eval_model = (
        Config.model_name
        if Config.model_name in {"ConvLSTM", "PureCNN", "TemporalTransformer"}
        else "TemporalTransformer"
    )
    parser.add_argument(
        "--model-name",
        default=default_eval_model,
        choices=["ConvLSTM", "PureCNN", "TemporalTransformer"],
        help="待评估的模型类型。",
    )
    parser.add_argument("--output-root", default=Config.output_root, help="统一结果根目录。")
    parser.add_argument(
        "--experiment-name",
        default=(Config.experiment_name or ""),
        help="实验名称；若为空，则自动使用模型对应的默认实验名。",
    )
    parser.add_argument("--checkpoint-path", default="", help="待评估权重路径；留空时按实验目录自动解析。")
    parser.add_argument("--output-dir", default="", help="评估结果输出目录；留空时默认写入实验的 results 目录。")
    parser.add_argument("--num-workers", default=Config.num_workers, type=int, help="DataLoader 的 worker 数量。")
    parser.add_argument("--batch-size", type=int, default=Config.batch_size, help="评估 batch size。")
    parser.add_argument("--test-augment", dest="test_augment", action="store_true", help="开启测试时数据增强。")
    parser.add_argument("--no-test-augment", dest="test_augment", action="store_false", help="关闭测试时数据增强。")
    parser.add_argument("--use-latest-checkpoint", action="store_true", help="当未显式传入 checkpoint 时，优先加载 latest_checkpoint.pth。")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="评估设备。")
    parser.set_defaults(test_augment=Config.test_augment)

    add_common_data_args(parser)
    add_convlstm_args(parser)
    add_transformer_args(parser)
    parser.add_argument(
        "--temporal-pool",
        choices=["mean", "max", "cls"],
        default=Config.temporal_pool,
        help="时间维聚合方式。ConvLSTM 支持 mean/max，Transformer 额外支持 cls。",
    )
    parser.add_argument("--temporal-stride", type=int, default=Config.temporal_stride, help="时间维下采样步长。")


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_train_cfg_from_args(args, model_name):
    base_cfg = Config()
    experiment_name = str(args.experiment_name or "").strip() or get_default_experiment_name(model_name)
    _, save_dir, results_dir, log_dir = build_experiment_dirs(args.output_root, experiment_name)

    overrides = {
        "model_name": model_name,
        "output_root": args.output_root,
        "data_root": args.data_root,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "frames": args.frames,
        "height": args.height,
        "width": args.width,
        "lidar_scale": args.lidar_scale,
        "max_hs": args.max_hs,
        "temporal_pool": args.temporal_pool,
        "temporal_stride": args.temporal_stride,
        "convlstm_hidden": getattr(args, "convlstm_hidden", base_cfg.convlstm_hidden),
        "convlstm_layers": getattr(args, "convlstm_layers", base_cfg.convlstm_layers),
        "vit_embed_dim": getattr(args, "vit_embed_dim", base_cfg.vit_embed_dim),
        "vit_depth": getattr(args, "vit_depth", base_cfg.vit_depth),
        "vit_num_heads": getattr(args, "vit_num_heads", base_cfg.vit_num_heads),
        "vit_mlp_ratio": getattr(args, "vit_mlp_ratio", base_cfg.vit_mlp_ratio),
        "vit_dropout": getattr(args, "vit_dropout", base_cfg.vit_dropout),
        "vit_attn_dropout": getattr(args, "vit_attn_dropout", base_cfg.vit_attn_dropout),
        "save_dir": save_dir,
        "results_dir": results_dir,
        "log_dir": log_dir,
        "resume_training": args.resume_training,
        "resume_checkpoint_path": args.resume_checkpoint_path,
        "save_latest_checkpoint": args.save_latest_checkpoint,
        "checkpoint_every": args.save_interval,
        "experiment_name": experiment_name,
        "log_interval": args.log_interval,
        "train_augment": not args.disable_train_augment,
        "seed": args.seed,
        "device": args.device,
    }
    return build_runtime_cfg(base_cfg, overrides)


def build_eval_cfg_from_args(args):
    base_cfg = Config()
    experiment_name = str(args.experiment_name or "").strip() or get_default_experiment_name(args.model_name)
    exp_root, save_dir, default_results_dir, log_dir = build_experiment_dirs(args.output_root, experiment_name)

    checkpoint_path = str(args.checkpoint_path or "").strip()
    if not checkpoint_path:
        checkpoint_name = "latest_checkpoint.pth" if args.use_latest_checkpoint else "best_model.pth"
        checkpoint_path = os.path.join(exp_root, "checkpoints", checkpoint_name)

    output_dir = str(args.output_dir or "").strip() or default_results_dir

    overrides = {
        "model_name": args.model_name,
        "output_root": args.output_root,
        "experiment_name": experiment_name,
        "data_root": args.data_root,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "frames": args.frames,
        "height": args.height,
        "width": args.width,
        "lidar_scale": args.lidar_scale,
        "max_hs": args.max_hs,
        "temporal_pool": args.temporal_pool,
        "temporal_stride": args.temporal_stride,
        "convlstm_hidden": args.convlstm_hidden,
        "convlstm_layers": args.convlstm_layers,
        "vit_embed_dim": args.vit_embed_dim,
        "vit_depth": args.vit_depth,
        "vit_num_heads": args.vit_num_heads,
        "vit_mlp_ratio": args.vit_mlp_ratio,
        "vit_dropout": args.vit_dropout,
        "vit_attn_dropout": args.vit_attn_dropout,
        "save_dir": save_dir,
        "results_dir": output_dir,
        "log_dir": log_dir,
        "test_augment": args.test_augment,
        "checkpoint_path": checkpoint_path,
        "device": args.device,
        "inference_use_latest_checkpoint": args.use_latest_checkpoint,
    }
    return build_runtime_cfg(base_cfg, overrides)
