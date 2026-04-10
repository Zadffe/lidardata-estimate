function [RangeTensor, LabelVector] = Generate_Clip(Hs, Tz, MainDir, Seed, settings)
    % 生成单个雷达波浪点云序列（一个 clip）。
    %
    % 这次修改的核心思想是：
    % 本函数内部的随机性必须与外层参数采样解耦。
    % 因此这里不再调用全局 rng(Seed)，
    % 而是改用局部随机流 local_stream。
    %
    % 好处是：
    % 1) 单个样本内部仍然可以通过 Seed 保证可复现
    % 2) 不会污染 Master_Data_Generator.m 外层的随机参数采样

    %% 1. 基础参数设置
    frames = round(settings.duration * settings.fps);
    dt = 1 / settings.fps;
    lidar.height = 15;

    % 雷达扫描角网格
    % Alpha_Grid / Beta_Grid 分别表示两个方向上的扫描角
    fov = 68;
    res = 0.68;
    angs = -fov/2 : res : fov/2;
    [Alpha_Grid, Beta_Grid] = meshgrid(deg2rad(angs), deg2rad(angs));
    [rows, cols] = size(Alpha_Grid);

    %% 2. 预计算雷达几何量
    % flat_Sx / flat_Sy 可理解为每条激光射线在 x/y 方向上的斜率项
    % 后面做射线与波面的交点迭代时会反复用到，所以先算好
    R_Sx = tan(Alpha_Grid);
    R_Sy = tan(Beta_Grid) ./ cos(Alpha_Grid);

    flat_Sx = gpuArray(single(R_Sx(:)));
    flat_Sy = gpuArray(single(R_Sy(:)));

    % Dist_Calm 表示“平静海面”下每条射线对应的距离
    % 后面会和真实波面下的距离做差，得到雷达量测变化量
    Dist_Calm = lidar.height * sqrt(1 + flat_Sx.^2 + flat_Sy.^2);

    %% 3. 构建二维方向波谱
    % 使用局部随机流，确保本函数内部随机性可复现，但不影响外层随机状态
    local_stream = RandStream('mt19937ar', 'Seed', double(Seed));

    g = 9.81;
    n_freq = 300;
    n_dir = 72;
    s_max = 25;

    % 根据 Tz 估计谱峰频率 wp，并设置采样频率范围
    wp = 2*pi / (Tz * 1.414);
    w_min = 0.5 * wp;
    w_max = 8.0 * wp;

    % 在频率区间内做带抖动的离散采样，避免过于规则的频率格点
    w_edges = linspace(w_min, w_max, n_freq + 1)';
    w_noise = rand(local_stream, n_freq, 1);
    w = w_edges(1:end-1) + w_noise .* diff(w_edges);
    dw = gradient(w);

    % 局部方向角 theta 只围绕主波向附近展开
    % 最后再统一加上 MainDir，得到全局传播方向
    theta_edges = linspace(-pi/2, pi/2, n_dir + 1);
    theta_noise = rand(local_stream, 1, n_dir);
    theta_local = theta_edges(1:end-1) + theta_noise .* diff(theta_edges);
    dtheta = gradient(theta_local);

    % 一维频谱 S(w) - ITTC 形式
    S_w = (173 * Hs^2 / Tz^4) ./ (w.^5) .* exp(-691 ./ (Tz^4 .* w.^4));

    % Mitsuyasu 方向扩展参数 s(w)
    S_spread_vec = zeros(n_freq, 1);
    for k = 1:n_freq
        if w(k) <= wp
            S_spread_vec(k) = s_max * (w(k) / wp)^5;
        else
            S_spread_vec(k) = s_max * (w(k) / wp)^(-2.5);
        end
    end

    % 构建二维频率-方向网格
    [W_grid, Theta_local_grid] = meshgrid(w, theta_local);

    % 方向分布 D(theta, w)
    S_param_grid = repmat(S_spread_vec', n_dir, 1);
    D_raw = max(0, cos(Theta_local_grid)).^(2 * S_param_grid);

    % 用数值积分做归一化，使得每个频率上的方向分布积分为 1
    dtheta_col = dtheta(:);
    dtheta_grid = repmat(dtheta_col, 1, n_freq);

    Integral_D = sum(D_raw .* dtheta_grid, 1);
    Norm_Factor = 1 ./ Integral_D;
    Norm_Grid = repmat(Norm_Factor, n_dir, 1);
    D_grid = D_raw .* Norm_Grid;

    % 二维方向波谱 S(w, theta)
    S_w_grid = repmat(S_w', n_dir, 1);
    Spec_2D = S_w_grid .* D_grid;

    % 由谱密度计算每个离散波分量的振幅
    dw_col = dw(:);
    dw_grid = repmat(dw_col', n_dir, 1);
    Amp = sqrt(2 * Spec_2D .* dw_grid .* dtheta_grid);

    % 做一次能量校正，使最终波高更贴近目标 Hs
    m0_sim = sum(Amp(:).^2) / 2;
    Hs_sim_raw = 4 * sqrt(m0_sim);
    if Hs_sim_raw > 0
        Amp = Amp * (Hs / Hs_sim_raw);
    end

    % 为每个离散波分量分配随机相位
    Phases = rand(local_stream, n_dir, n_freq) * 2 * pi;

    % 把局部方向角转成全局方向角
    Theta_global = Theta_local_grid + MainDir;
    K_num = (W_grid.^2) / g;
    Kx = K_num .* cos(Theta_global);
    Ky = K_num .* sin(Theta_global);

    % 将波谱参数转到 GPU
    p_amp = gpuArray(single(Amp(:)));
    p_kx  = gpuArray(single(Kx(:)));
    p_ky  = gpuArray(single(Ky(:)));
    p_w   = gpuArray(single(W_grid(:)));
    p_phi = gpuArray(single(Phases(:)));

    %% 4. 在 GPU 上逐帧生成波浪场，并模拟雷达扫描
    RangeTensor_GPU = zeros(frames, rows, cols, 'single', 'gpuArray');

    % 构造真实波面所在的大范围平面网格
    % 先在这个规则网格上生成波面，再插值得到每条雷达射线上的交点高度
    roi_radius = lidar.height * tan(deg2rad(fov / 2)) + 8;
    sim_res = 0.2;

    x_vec = -roi_radius:sim_res:roi_radius;
    y_vec = -roi_radius:sim_res:roi_radius;
    [Grid_X, Grid_Y] = meshgrid(x_vec, y_vec);

    Grid_X = gpuArray(single(Grid_X));
    Grid_Y = gpuArray(single(Grid_Y));
    flat_Grid_X = Grid_X(:)';
    flat_Grid_Y = Grid_Y(:)';

    % 波分量很多，分块累加可以避免显存过高
    block_size = 5000;
    num_components = length(p_amp);

    for t = 1:frames
        cur_time = (t - 1) * dt;

        % Z_flat 表示当前时刻规则网格上的波面高度
        Z_flat = zeros(1, length(flat_Grid_X), 'single', 'gpuArray');

        for k = 1:block_size:num_components
            idx_end = min(k + block_size - 1, num_components);
            idx_range = k:idx_end;

            % 逐块计算离散波分量在整张网格上的贡献
            ST_block = p_kx(idx_range) * flat_Grid_X + p_ky(idx_range) * flat_Grid_Y;
            phase_block = p_phi(idx_range) - p_w(idx_range) * cur_time;
            Z_flat = Z_flat + sum(p_amp(idx_range) .* cos(ST_block + phase_block), 1);
        end

        Grid_Z = reshape(Z_flat, size(Grid_X));

        % 对每条雷达射线做 3 次固定点迭代，求其与真实波面的交点高度
        Z_est = zeros(length(flat_Sx), 1, 'single', 'gpuArray');
        for iter = 1:3
            X_q = (lidar.height - Z_est) .* flat_Sx;
            Y_q = (lidar.height - Z_est) .* flat_Sy;
            Z_est = interp2(Grid_X, Grid_Y, Grid_Z, X_q, Y_q, 'linear');
        end

        % Dist_Wave: 真实波面下的传播距离
        % Delta_Range: 平静海面与真实波面之间的距离差
        % 最终保存的就是这个“量测变化张量”
        Dist_Wave = (lidar.height - Z_est) .* sqrt(1 + flat_Sx.^2 + flat_Sy.^2);
        Delta_Range = Dist_Calm - Dist_Wave;
        Delta_Range(isnan(Delta_Range)) = 0;

        RangeTensor_GPU(t, :, :) = reshape(Delta_Range, rows, cols);
    end

    %% 5. 搬回 CPU，并组织标签
    RangeTensor = gather(RangeTensor_GPU);

    % 标签设计：
    % 方向用 sin / cos 编码，避免 0° 和 360° 的跳变问题
    % Hs 直接作为第三个标签分量
    LabelVector = [sin(MainDir), cos(MainDir), Hs];
end
