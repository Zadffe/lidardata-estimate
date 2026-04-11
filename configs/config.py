import os


class Config:
    data_root = r"F:\Research__dir\dl_lidar\datasets\Dataset_Wave_Lidar_v3_10000samples"
    output_root = "./all_exps_result"

    epochs = 300
    batch_size = 32
    lr = 1e-4
    weight_decay = 1e-4
    num_workers = 4
    log_interval = 20
    save_interval = 20
    seed = 42

    frames = 64
    height = 101
    width = 101
    max_hs = 6.0
    lidar_scale = 5.0

    # 可选模型名称：
    # - "ConvLSTM"
    # - "PureCNN"
    # - "TemporalTransformer"
    model_name = "TemporalTransformer"
    pretrained = False

    # 当 experiment_name 为空时，
    # 默认使用 "<标准化模型名>_<experiment_tag>" 作为实验目录名。
    experiment_tag = "datasetsv3_realdataloss_80drop"
    experiment_name = ""

    resume_training = False
    resume_checkpoint_path = ""
    save_latest_checkpoint = True
    checkpoint_every = 20

    inference_use_latest_checkpoint = False
    inference_checkpoint_path = ""

    test_augment = True

    convlstm_hidden = 64
    convlstm_layers = 1

    temporal_pool = "max"
    temporal_stride = 2

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
        run_name = str(getattr(self, "experiment_name", "") or "").strip()
        if not run_name:
            run_name = f"{model_dir}_{self.experiment_tag}"

        self.save_dir = os.path.join(self.output_root, run_name, "checkpoints")
        self.results_dir = os.path.join(self.output_root, run_name, "results")
        self.log_dir = os.path.join(self.output_root, run_name, "logs")
