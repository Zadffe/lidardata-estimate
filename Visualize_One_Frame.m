function Visualize_One_Frame(InputData, frame_idx)
    % VISUALIZE_ONE_FRAME (智能版)
    % 输入: 
    %   InputData: 可以是生成的矩阵 [Frames, Rows, Cols]
    %              也可以是 .mat 文件的路径字符串 (例如 'train/Sample_01.mat')
    %   frame_idx: 想看第几帧 (默认第 10 帧)
    
    if nargin < 2
        frame_idx = 10; % 默认看第10帧
    end

    % 加载数据

    if ischar(InputData) || isstring(InputData)
        % 如果输入是字符串，说明是文件路径，自动加载
        fprintf('正在加载文件: %s ...\n', InputData);
        loaded_struct = load(InputData);
        
        % 尝试寻找其中的数据矩阵
        % 在 Master Generator 中我们保存为 'tensor'
        if isfield(loaded_struct, 'tensor')
            RangeTensor = loaded_struct.tensor;
        elseif isfield(loaded_struct, 'RangeTensor')
            RangeTensor = loaded_struct.RangeTensor;
        elseif isfield(loaded_struct, 'WaveVideo')
            RangeTensor = loaded_struct.WaveVideo;
        else
            error('在 .mat 文件中未找到 tensor/RangeTensor/WaveVideo 变量。请检查变量名。');
        end
        
        % 顺便打印一下这个文件的标签信息
        if isfield(loaded_struct, 'labels')
            labs = loaded_struct.labels;
            % labels: [sin, cos, Hs]
            % 计算角度
            ang = atan2(labs(1), labs(2)) * (180/pi);
            if ang < 0, ang = ang + 360; end
            fprintf('文件标签 -> Hs: %.2fm, 方向: %.1f度\n', labs(3), ang);
        end
    else
        % 如果输入已经是矩阵
        RangeTensor = InputData;
    end
    
    % =============================================================
    % 2. 提取帧数据
    % =============================================================
    % 检查帧数是否越界
    num_frames = size(RangeTensor, 1);
    if frame_idx > num_frames
        warning('请求的帧数 (%d) 超过总帧数 (%d)，已自动调整为最后一帧。', frame_idx, num_frames);
        frame_idx = num_frames;
    end

    img = squeeze(RangeTensor(frame_idx, :, :));
    
    % 雷达参数 (必须与生成时一致)
    fov = 68;       
    res = 0.68;     
    h_lidar = 15;   
    
    % 重建角度网格
    angs = -fov/2 : res : fov/2; 
    [Alpha, Beta] = meshgrid(deg2rad(angs), deg2rad(angs));
    
    % =============================================================
    % 3. 绘图 (Range View)
    % =============================================================
    figure('Color', 'white', 'Position', [100, 100, 1200, 500]);
    
    subplot(1, 2, 1);
    imagesc(img);
    colormap jet; colorbar;
    title(sprintf('CNN Input (Range View) - Frame %d', frame_idx));
    xlabel('Azimuth Grid'); ylabel('Elevation Grid');
    axis square;
    
    % =============================================================
    % 4. 绘图 (3D 重构)
    % =============================================================
    % 反算几何坐标
    Sx = tan(Alpha);
    Sy = tan(Beta) ./ cos(Alpha);
    Norm = sqrt(1 + Sx.^2 + Sy.^2);
    
    Dist_Calm = h_lidar * Norm;
    Dist_Meas = Dist_Calm - img; % 还原出测量距离
    
    Z_water = h_lidar - (Dist_Meas ./ Norm);
    X_world = (h_lidar - Z_water) .* Sx;
    Y_world = (h_lidar - Z_water) .* Sy;
    
    subplot(1, 2, 2);
    surf(X_world, Y_world, Z_water, 'EdgeColor', 'none');
    shading interp;
    
    % 灯光效果让波浪更明显
    light('Position',[-1 -1 1],'Style','infinite');
    lighting gouraud;
    material shiny; % 让水面有点反光感
    
    axis equal; grid on;
    title('Reconstructed 3D Wave Surface');
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    view(3); 
    zlim([-6, 6]); % 固定Z轴范围方便对比
end