import numpy as np
import torch

def process_real_point_cloud(point_cloud_frames, cfg):
    """
    将真实杂乱的点云转换为神经网络需要的 Tensor。
    
    Args:
        point_cloud_frames: 一个列表，包含几十帧数据。
                            每一帧是一个 (N, 3) 的 numpy 数组 [x, y, z]。
                            注意：N 可以是变化的（有些帧点多，有些点少）。
        cfg: 配置对象，包含 FOV, grid_size 等参数。
        
    Returns:
        input_tensor: [1, 1, T, H, W] 的 PyTorch 张量
    """
    T = cfg.frames
    H, W = cfg.height, cfg.width # 比如 101
    fov = 68.0 # 你的雷达视场角
    
    # 初始化输出容器 (全0，相当于默认就是空缺/盲区)
    # 这就解决了"空缺点云"的问题，空的地方自然就是0
    tensor_volume = np.zeros((T, H, W), dtype=np.float32)
    
    # 预计算角度边界
    min_ang = -fov / 2
    max_ang = fov / 2
    res = fov / (H - 1) # 角度分辨率
    
    print(f"开始转换实测数据: 共 {len(point_cloud_frames)} 帧...")
    
    for t in range(min(T, len(point_cloud_frames))):
        pts = point_cloud_frames[t] # 取出一帧: [N, 3]
        
        if pts is None or len(pts) == 0:
            continue # 如果这一帧完全丢失，就留着全0，模型能处理
            
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        
        # --- 步骤 A: 坐标系转换 (笛卡尔 -> 球坐标索引) ---
        # 1. 计算每个点的水平角(Azimuth)和垂直角(Elevation)
        # 注意: 这里的公式要和你 MATLAB 生成时的几何逻辑一致
        # 假设 x向前, y向左, z向上 (需根据你实际雷达安装调整)
        
        # 计算距离 (Range)
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # 计算角度 (弧度)
        # 这里是一个示例投影，具体取决于你雷达是机械旋转还是MEMS扫描
        # 假设是 MEMS 矩形扫描:
        # alpha = arctan(y/x), beta = arcsin(z/r) (仅做示例)
        
        # 为了对应你之前的 MATLAB 逻辑 (Tan 投影):
        # 你的 MATLAB 逻辑是: Sx = x/z_depth, Sy = y/z_depth
        # 或者是基于角度网格。
        # 最稳妥的方法是直接利用 x, y 坐标映射，因为你在 20m x 20m 范围内
        
        # --- 简单方案: 基于 X, Y 物理坐标的网格化 (更鲁棒) ---
        # 如果你的训练数据本质上是 20m x 20m 的范围
        roi = 20.0
        
        # 计算网格索引 (0 ~ 100)
        # 将 x 从 [-10, 10] 映射到 [0, 100]
        grid_x = ((x + roi/2) / roi * (W - 1)).astype(int)
        grid_y = ((y + roi/2) / roi * (H - 1)).astype(int)
        
        # 过滤掉超出范围的点
        valid_mask = (grid_x >= 0) & (grid_x < W) & \
                     (grid_y >= 0) & (grid_y < H)
        
        grid_x = grid_x[valid_mask]
        grid_y = grid_y[valid_mask]
        val_z  = z[valid_mask] # 取高度作为特征，或者取 Range
        
        # --- 步骤 B: 填值 (解决杂乱和冲突) ---
        # 如果多个点落入同一个格子，最简单是取最后覆盖，或者取平均
        # 这里用一种快速方法:
        tensor_volume[t, grid_y, grid_x] = val_z
        
        # 如果你想取平均（更精确），可以用 numpy 的 accumarray 类似逻辑
        # 但通常直接覆盖对于推理来说足够了
        
    # --- 步骤 C: 归一化 ---
    # 这一步必须做！要和训练时保持一致
    tensor_volume = tensor_volume / cfg.lidar_scale 
    
    # 转 Tensor
    input_tensor = torch.from_numpy(tensor_volume).unsqueeze(0).unsqueeze(0)
    # [1, 1, T, H, W]
    
    return input_tensor.float()