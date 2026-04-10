class Config:
    data_root = r"F:\Research__dir\dl_lidar\datasets\Dataset_Wave_Lidar_v3"
    save_dir = "./checkpoints/ConvLSTM_datasetsv3_realdataloss"
    results_dir = "./results/ConvLSTM_datasetsv3_realdataloss"
    log_dir = "./logs/ConvLSTM_datasetsv3_realdataloss"

    epochs = 100
    batch_size = 32
    lr = 1e-4
    weight_decay = 1e-4
    num_workers = 4

    frames = 64
    height = 101
    width = 101
    max_hs = 6.0
    lidar_scale = 5.0

    # 可选:
    # - "ConvLSTM"
    # - "PureCNN"
    # - "TemporalTransformer"
    model_name = "ConvLSTM"
    pretrained = False

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
