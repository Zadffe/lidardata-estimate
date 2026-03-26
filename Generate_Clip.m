function [RangeTensor, LabelVector] = Generate_Clip(Hs, Tz, MainDir, Seed, settings)
    % 依赖Parallel Computing Toolbox工具箱, NVIDIA GPU显卡
    
    %% 1. 参数设置与几何初始化
    frames = round(settings.duration * settings.fps);
    dt = 1 / settings.fps;
    lidar.height = 15;
    
    % 扫描角度网格
    fov = 68; res = 0.68;
    angs = -fov/2 : res : fov/2; 
    [Alpha_Grid, Beta_Grid] = meshgrid(deg2rad(angs), deg2rad(angs));
    [rows, cols] = size(Alpha_Grid);
    
    %% 2. 预计算几何关系
    % 这里的计算量很小，先在 CPU 算好，最后转 GPU
    R_Sx = tan(Alpha_Grid); 
    R_Sy = tan(Beta_Grid) ./ cos(Alpha_Grid); 
    
    % 转为 GPU 数组
    flat_Sx = gpuArray(single(R_Sx(:)));
    flat_Sy = gpuArray(single(R_Sy(:)));
    
    Dist_Calm = lidar.height * sqrt(1 + flat_Sx.^2 + flat_Sy.^2); % GPU Array
    
    %% 3. 波浪谱生成 (完全采用你提供的 ITTC 谱与抖动采样逻辑)
    rng(Seed); 
    g = 9.81;
    n_freq = 300; 
    n_dir = 72;
    s_max = 25; 
    
    % --- A. 频率与方向向量生成 (带抖动采样) ---
    wp = 2*pi / (Tz * 1.414); 
    w_min = 0.5 * wp;
    w_max = 8.0 * wp; 
    
    % 频率 w
    w_edges = linspace(w_min, w_max, n_freq + 1)';
    w_noise = rand(n_freq, 1); 
    w = w_edges(1:end-1) + w_noise .* diff(w_edges);
    dw = gradient(w); % [n_freq, 1]
    
    % 方向 theta (使用局部相对角度 -pi/2 到 pi/2，保证主波向在中心)
    theta_edges = linspace(-pi/2, pi/2, n_dir + 1);
    theta_noise = rand(1, n_dir);
    theta_local = theta_edges(1:end-1) + theta_noise .* diff(theta_edges);
    dtheta = gradient(theta_local); % [1, n_dir] 行向量
    
    % --- B. 2D 谱密度计算 (S_2D) ---
    % 1. 频率谱 S(w) - ITTC 1978
    S_w = (173 * Hs^2 / Tz^4) ./ (w.^5) .* exp(-691 ./ (Tz^4 .* w.^4));
    
    % 2. 方向分布参数 s(w) - Mitsuyasu 分段
    S_spread_vec = zeros(n_freq, 1);
    for k = 1:n_freq
        if w(k) <= wp
            S_spread_vec(k) = s_max * (w(k)/wp)^5;
        else
            S_spread_vec(k) = s_max * (w(k)/wp)^(-2.5);
        end
    end
    
    % 3. 构建网格 [n_dir, n_freq]
    [W_grid, Theta_local_grid] = meshgrid(w, theta_local);
    
    % 4. 计算方向分布 D(theta, w)
    S_param_grid = repmat(S_spread_vec', n_dir, 1);
    % 在相对坐标系下，主方向差值就是 Theta_local_grid 本身
    D_raw = max(0, cos(Theta_local_grid)).^(2 * S_param_grid);
    
    % 归一化计算
    dtheta_col = dtheta(:); 
    dtheta_grid = repmat(dtheta_col, 1, n_freq); % [n_dir, n_freq]
    
    Integral_D = sum(D_raw .* dtheta_grid, 1); 
    Norm_Factor = 1 ./ Integral_D; 
    Norm_Grid = repmat(Norm_Factor, n_dir, 1);
    D_grid = D_raw .* Norm_Grid;
    
    % 5. 合成 2D 谱 S(w, theta)
    S_w_grid = repmat(S_w', n_dir, 1);
    Spec_2D = S_w_grid .* D_grid;
    
    % --- C. 计算振幅与相位 ---
    dw_col = dw(:);
    dw_grid = repmat(dw_col', n_dir, 1); % [n_dir, n_freq]
    
    % A = sqrt(2 * S(w,th) * dw * dtheta)
    Amp = sqrt(2 * Spec_2D .* dw_grid .* dtheta_grid);
    
    % 能量校正
    m0_sim = sum(Amp(:).^2) / 2;
    Hs_sim_raw = 4 * sqrt(m0_sim);
    if Hs_sim_raw > 0
        Amp = Amp * (Hs / Hs_sim_raw);
    end
    
    % 随机相位
    Phases = rand(n_dir, n_freq) * 2 * pi;
    
    % 转换到全局绝对角度并计算波数 K
    Theta_global = Theta_local_grid + MainDir;
    K_num = (W_grid.^2) / g;
    Kx = K_num .* cos(Theta_global);
    Ky = K_num .* sin(Theta_global);
    
    % 展平参数并送入 GPU
    p_amp = gpuArray(single(Amp(:))); 
    p_kx  = gpuArray(single(Kx(:))); 
    p_ky  = gpuArray(single(Ky(:))); 
    p_w   = gpuArray(single(W_grid(:))); 
    p_phi = gpuArray(single(Phases(:)));
    
    %% 4. 利用GPU进行加速计算 (分块极速版)
    RangeTensor_GPU = zeros(frames, rows, cols, 'single', 'gpuArray');
    
    % 网格生成设置 (用于真值表面)
    roi_radius = lidar.height * tan(deg2rad(fov/2)) + 8; 
    sim_res = 0.2; % 采用你之前代码中的 0.2，避免矩阵过大且能保留纹理
    
    x_vec = -roi_radius:sim_res:roi_radius;
    y_vec = -roi_radius:sim_res:roi_radius;
    [Grid_X, Grid_Y] = meshgrid(x_vec, y_vec);
    
    Grid_X = gpuArray(single(Grid_X));
    Grid_Y = gpuArray(single(Grid_Y));
    flat_Grid_X = Grid_X(:)'; 
    flat_Grid_Y = Grid_Y(:)';
    
    for t = 1:frames
        cur_time = (t-1) * dt;
        
        % --- A. 计算真值波面 Z (GPU 分块计算，内存安全) ---
        Z_flat = zeros(1, length(flat_Grid_X), 'single', 'gpuArray');
        
        % 设置分块大小防止显存溢出 (4070 Ti SUPER 跑 5000 毫无压力)
        block_size = 5000; 
        num_components = length(p_amp);
        
        for k = 1:block_size:num_components
            idx_end = min(k + block_size - 1, num_components);
            idx_range = k:idx_end;
            
            % 计算空间项和相位
            ST_block = p_kx(idx_range) * flat_Grid_X + p_ky(idx_range) * flat_Grid_Y; 
            phase_block = p_phi(idx_range) - p_w(idx_range) * cur_time;
            
            % 累加到 Z 平面
            Z_flat = Z_flat + sum(p_amp(idx_range) .* cos(ST_block + phase_block), 1);
        end
        
        Grid_Z = reshape(Z_flat, size(Grid_X));
        
        % --- B. 射线追踪 (不动点迭代) ---
        Z_est = zeros(length(flat_Sx), 1, 'single', 'gpuArray');
        for iter = 1:3
            X_q = (lidar.height - Z_est) .* flat_Sx;
            Y_q = (lidar.height - Z_est) .* flat_Sy;
            
            Z_new = interp2(Grid_X, Grid_Y, Grid_Z, X_q, Y_q, 'linear');
            Z_est = Z_new;
        end
        
        % --- C. 计算特征 ---
        Dist_Wave = (lidar.height - Z_est) .* sqrt(1 + flat_Sx.^2 + flat_Sy.^2);
        Delta_Range = Dist_Calm - Dist_Wave;
        Delta_Range(isnan(Delta_Range)) = 0;
        
        RangeTensor_GPU(t, :, :) = reshape(Delta_Range, rows, cols);
    end
    
    %% 5. 将结果搬回 CPU 并生成标签
    RangeTensor = gather(RangeTensor_GPU);
    LabelVector = [sin(MainDir), cos(MainDir), Hs];
end