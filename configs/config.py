class Config:
    data_root = r"F:\Research__dir\dl_lidar\datasets\Dataset_Wave_Lidar_v2"
    save_dir = "./checkpoints/ConvLSTM_dataloss_datasetsv2_3channel_train_no_noise"
    results_dir = "./results/ConvLSTM_dataloss_datasetsv2_3channel_train_no_noise"
    log_dir = "./logs/ConvLSTM_dataloss_datasetsv2_3channel_train_no_noise"

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

    model_name = "ConvLSTM"
    pretrained = False
    convlstm_hidden = 64
    convlstm_layers = 1
    temporal_pool = "max"
    temporal_stride = 2
