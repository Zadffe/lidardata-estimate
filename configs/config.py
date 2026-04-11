class Config:
    data_root = r"F:\Research__dir\dl_lidar\datasets\Dataset_Wave_Lidar_v3"

    epochs = 300
    batch_size = 32
    lr = 1e-4
    weight_decay = 1e-4
    num_workers = 4

    frames = 64
    height = 101
    width = 101
    max_hs = 6.0
    lidar_scale = 5.0

    # 可选模型:
    # - "ConvLSTM"
    # - "PureCNN"
    # - "TemporalTransformer"
    model_name = "TemporalTransformer"
    pretrained = False

    # 实验名称后缀。
    # 输出目录会自动生成为:
    # checkpoints/<模型名>_<experiment_tag>
    # results/<模型名>_<experiment_tag>
    # logs/<模型名>_<experiment_tag>
    experiment_tag = "datasetsv3_realdataloss_80drop"

    # 断点续训配置
    resume_training = False
    resume_checkpoint_path = ""
    save_latest_checkpoint = True
    checkpoint_every = 20

    # 推理 / 评估 checkpoint 配置
    # 默认读取 best_model.pth
    # 如果 inference_use_latest_checkpoint=True，则优先读取 latest_checkpoint.pth
    # 如果 inference_checkpoint_path 非空，则优先读取这个显式路径
    inference_use_latest_checkpoint = False
    inference_checkpoint_path = ""

    # 评估 / 推理时是否对测试集启用增强
    test_augment = True

    # CNN + ConvLSTM 参数
    convlstm_hidden = 64
    convlstm_layers = 1

    # 公共时序参数
    temporal_pool = "max"
    temporal_stride = 2

    # CNN Stem + Temporal Transformer 参数
    vit_embed_dim = 128
    vit_depth = 4
    vit_num_heads = 8
    vit_mlp_ratio = 4.0
    vit_dropout = 0.1
    vit_attn_dropout = 0.1

    def __init__(self):
        self.refresh_output_dirs()

    @staticmethod
    def _normalize_model_dir_name(model_name: str) -> str:
        name = str(model_name).strip().lower()

        if name in ("convlstm", "cnn_convlstm", "cnn+convlstm"):
            return "ConvLSTM"
        if name in ("purecnn", "cnn", "cnn_only", "pure_cnn"):
            return "CNN"
        if name in (
            "temporaltransformer",
            "cnn_temporal_transformer",
            "cnn+transformer",
            "transformer",
            "vit",
        ):
            return "TemporalTransformer"

        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(model_name))
        return safe_name or "UnknownModel"

    def refresh_output_dirs(self):
        model_dir = self._normalize_model_dir_name(self.model_name)
        run_name = f"{model_dir}_{self.experiment_tag}"

        self.save_dir = f"./checkpoints/{run_name}"
        self.results_dir = f"./results/{run_name}"
        self.log_dir = f"./logs/{run_name}"
