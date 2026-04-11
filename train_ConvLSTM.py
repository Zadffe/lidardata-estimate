import argparse

from configs.config import Config
from utils.runtime_utils import get_default_experiment_name
from utils.training_utils import run_training


def create_parser():
    parser = argparse.ArgumentParser(
        description="ConvLSTM 模型训练脚本。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output-root", type=str, default=Config.output_root, help="统一结果根目录。")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=(Config.experiment_name or get_default_experiment_name("ConvLSTM")),
        help="实验名称，对应输出目录名。",
    )
    parser.add_argument("--epochs", type=int, default=Config.epochs, help="训练轮数。")
    parser.add_argument("--batch-size", type=int, default=Config.batch_size, help="训练 batch size。")
    parser.add_argument("--lr", type=float, default=Config.lr, help="学习率。")
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay, help="AdamW 的 weight decay。")
    parser.add_argument("--num-workers", type=int, default=Config.num_workers, help="DataLoader 的 worker 数量。")
    parser.add_argument("--log-interval", type=int, default=Config.log_interval, help="每隔多少个 batch 打印一次日志。")
    parser.add_argument("--save-interval", type=int, default=Config.save_interval, help="每隔多少个 epoch 保存一次阶段性权重。")
    parser.add_argument("--seed", type=int, default=Config.seed, help="随机种子。")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="训练设备。")

    parser.add_argument("--data-root", type=str, default=Config.data_root, help="数据集根目录。")
    parser.add_argument("--frames", type=int, default=Config.frames, help="每个样本包含的时间帧数。")
    parser.add_argument("--height", type=int, default=Config.height, help="输入张量高度。")
    parser.add_argument("--width", type=int, default=Config.width, help="输入张量宽度。")
    parser.add_argument("--lidar-scale", type=float, default=Config.lidar_scale, help="LiDAR 高度值归一化尺度。")
    parser.add_argument("--max-hs", type=float, default=Config.max_hs, help="波高 Hs 的反归一化上限。")

    parser.add_argument("--temporal-pool", type=str, default=Config.temporal_pool, choices=["mean", "max"], help="时间维聚合方式。")
    parser.add_argument("--temporal-stride", type=int, default=Config.temporal_stride, help="时间维下采样步长。")
    parser.add_argument("--convlstm-hidden", type=int, default=Config.convlstm_hidden, help="ConvLSTM 隐藏通道数。")
    parser.add_argument("--convlstm-layers", type=int, default=Config.convlstm_layers, help="ConvLSTM 层数。")

    parser.add_argument("--resume-training", action="store_true", help="从默认 latest_checkpoint.pth 继续训练。")
    parser.add_argument("--resume-checkpoint-path", type=str, default="", help="从指定 checkpoint 路径继续训练。")
    parser.add_argument("--save-latest-checkpoint", dest="save_latest_checkpoint", action="store_true", help="训练过程中保存 latest_checkpoint.pth。")
    parser.add_argument("--no-save-latest-checkpoint", dest="save_latest_checkpoint", action="store_false", help="关闭 latest_checkpoint.pth 的保存。")
    parser.add_argument("--disable-train-augment", action="store_true", help="关闭训练集数据增强。")
    parser.set_defaults(save_latest_checkpoint=Config.save_latest_checkpoint)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    run_training("ConvLSTM", args)


if __name__ == "__main__":
    main()
