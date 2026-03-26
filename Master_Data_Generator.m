
clc; clear; close all;

%% 1. 环境检查
if gpuDeviceCount("available") < 1
    error('错误: 未检测到支持 CUDA 的 GPU 设备！请检查显卡驱动或使用 CPU 版代码。');
else
    g = gpuDevice(1);
    fprintf('=== 检测到 GPU 加速环境 ===\n');
    fprintf('设备名称: %s\n', g.Name);
    fprintf('可用显存: %.2f GB\n', g.AvailableMemory / 1e9);
    fprintf('===========================\n');
    % 重置 GPU 以清空之前的残留内存
    reset(g); 
end

%% 2. 配置
base_dir = 'Dataset_Wave_Lidar_v2'; % 建议换个文件夹名区分
n_train = 2100;  % GPU生成很快，你可以尝试设大一点
n_val   = 450;
n_test  = 450;

clip_settings.duration = 16;       
clip_settings.fps = 4;             
clip_settings.grid_size = 101;     
clip_settings.roi = 20;            

range.Hs = [1.0, 5.5];             
range.Tz = [4.0, 8.0];             
range.Dir = [0, 2*pi];             

%% 3. 初始化目录
folders = {fullfile(base_dir, 'train'), fullfile(base_dir, 'val'), fullfile(base_dir, 'test')};
for i = 1:length(folders)
    if ~exist(folders{i}, 'dir'), mkdir(folders{i}); end
end

tasks = {n_train, folders{1}, 'Train'; n_val, folders{2}, 'Val'; n_test, folders{3}, 'Test'};

%% 4. 主生成循环
total_samples = n_train + n_val + n_test;
global_counter = 0;
start_time = tic;

for k = 1:size(tasks, 1)
    num_samples = tasks{k, 1};
    save_path   = tasks{k, 2};
    task_name   = tasks{k, 3};
    
    if num_samples == 0, continue; end
    
    fprintf('\n=== 开始生成 %s 集 (共 %d 个样本) ===\n', task_name, num_samples);
    
    % 注意：这里使用普通的 for 循环，不要用 parfor
    for i = 1:num_samples
        
        % 1. 随机参数
        this_Hs = range.Hs(1) + rand * (range.Hs(2) - range.Hs(1));
        
        min_T = sqrt(this_Hs) * 2.5; 
        actual_min_T = max(min_T, range.Tz(1));
        this_Tz = actual_min_T + rand * (range.Tz(2) - actual_min_T);
        
        this_Dir = range.Dir(1) + rand * (range.Dir(2) - range.Dir(1));
        
        % 随机种子
        this_Seed = round(rand * 1000000); 
        
        % 2. 调用 GPU 核心函数
        [tensor_data, labels] = Generate_Clip(this_Hs, this_Tz, this_Dir, this_Seed, clip_settings);


        data_to_save = struct();
        data_to_save.tensor = tensor_data;       % [Frames, H, W]
        data_to_save.labels = labels;       % [sin, cos, Hs]
        data_to_save.params = [this_Hs, this_Tz, this_Dir]; % [Hs, Tz, Dir]
        
        % 3. 保存 (Tensor 已经是 gather 过的 CPU 数组，直接保存即可)
        fname = sprintf('Sample_%04d_Hs%.1f_Tz%.1f_Dirdeg%.0f.mat', ...
            i, this_Hs, this_Tz, rad2deg(this_Dir));
        full_path = fullfile(save_path, fname);
        
        % 直接保存，不需要 par_save 辅助函数
        save(full_path, '-struct', 'data_to_save');
        
        % 4. 进度显示 (GPU 版直接打印即可)
        global_counter = global_counter + 1;
        
        % 每 10 个样本打印一次进度，避免刷屏
        if mod(i, 10) == 0 || i == num_samples
            elapsed = toc(start_time);
            avg_time = elapsed / global_counter;
            remain_time = avg_time * (total_samples - global_counter);
            
            fprintf('[%s] 进度: %d/%d | 总耗时: %.1fs | 速度: %.2f秒/个 | 剩余: %.0fs\n', ...
                task_name, i, num_samples, elapsed, avg_time, remain_time);
        end
    end
end

fprintf('\n所有数据生成完毕！保存于: %s\n', base_dir);