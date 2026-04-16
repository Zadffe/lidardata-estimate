clc; clear; close all;

%% 1. 运行环境检查
% 检查是否存在可用 GPU。
% 这个数据生成流程依赖 GPU 来加速波浪场与雷达扫描模拟。
if gpuDeviceCount("available") < 1
    error('未检测到支持 CUDA 的 GPU。');
else
    g = gpuDevice(1);
    fprintf('=== 检测到 GPU ===\n');
    fprintf('设备名称: %s\n', g.Name);
    fprintf('可用显存: %.2f GB\n', g.AvailableMemory / 1e9);
    fprintf('==================\n');
    reset(g);
end

%% 2. 全局配置
% base_dir: 数据集输出目录
% n_train / n_val / n_test: 三个划分中样本数量
base_dir = '../datasets/Dataset_Wave_Lidar_v3_10000samples';
n_train = 8000;
n_val   = 1000;
n_test  = 1000;

% master_seed 只控制“参数采样与划分”。
% 这样每次重新生成数据时，只要这个种子不变，参数池和 split 就可复现。
master_seed = 20260410;

% clip_settings 是单个样本（一个 clip）的基础设置。
clip_settings.duration = 16;
clip_settings.fps = 4;
clip_settings.grid_size = 101;
clip_settings.roi = 20;

% 波浪参数取值范围
range.Hs = [1.0, 5.5];
range.Tz = [4.0, 8.0];
range.Dir = [0, 2*pi];

%% 3. 准备输出文件夹
folders = {
    fullfile(base_dir, 'train'), ...
    fullfile(base_dir, 'val'), ...
    fullfile(base_dir, 'test')
};

for i = 1:numel(folders)
    if ~exist(folders{i}, 'dir')
        mkdir(folders{i});
    end
end

% tasks 中每一行分别对应一个数据划分：
% [样本数, 保存路径, 名称]
tasks = {
    n_train, folders{1}, 'Train'; ...
    n_val,   folders{2}, 'Val'; ...
    n_test,  folders{3}, 'Test'
};

%% 4. 先生成“唯一参数池”，再划分 train / val / test
% 这是这次修改最关键的部分。
%
% 旧逻辑的问题：
% 边采样参数、边生成样本、边写入 train/val/test，
% 再加上 Generate_Clip 内部会改写全局随机数状态，
% 容易导致不同 split 之间出现重复参数组合。
%
% 新逻辑：
% 1) 先在外层一次性采样出 total_samples 个“唯一”的 (Hs, Tz, Dir)
% 2) 再打乱这些参数
% 3) 最后再切分到 train / val / test
%
% 这样可以从逻辑上保证三个数据划分不会共享同一组参数组合。
total_samples = n_train + n_val + n_test;
rng(master_seed, 'twister');

% param_pool 用来存储所有待生成样本的参数。
% 每个元素记录：
% Hs / Tz / Dir / Seed
% 其中 Seed 是给 Generate_Clip 内部随机过程使用的局部种子。
param_pool(total_samples) = struct('Hs', 0, 'Tz', 0, 'Dir', 0, 'Seed', 0);

% used_keys 用于显式判重，避免出现重复参数组合。
% key 的格式由 make_param_key() 决定。
used_keys = containers.Map('KeyType', 'char', 'ValueType', 'logical');

sample_idx = 1;
attempt_count = 0;
max_attempts = total_samples * 500;

fprintf('\n开始生成 %d 组唯一参数组合...\n', total_samples);

while sample_idx <= total_samples
    attempt_count = attempt_count + 1;
    if attempt_count > max_attempts
        error('唯一参数组合采样失败：尝试次数超过上限。');
    end

    % 1) 随机采样 Hs
    this_Hs = range.Hs(1) + rand() * (range.Hs(2) - range.Hs(1));

    % 2) 根据经验关系约束 Tz 的下界
    %    这里保留你原来的逻辑：Hs 越大，Tz 的合理下限越高
    min_T = sqrt(this_Hs) * 2.5;
    actual_min_T = max(min_T, range.Tz(1));
    this_Tz = actual_min_T + rand() * (range.Tz(2) - actual_min_T);

    % 3) 随机采样主波向
    this_Dir = range.Dir(1) + rand() * (range.Dir(2) - range.Dir(1));

    % 4) 生成一个给单条样本内部波场使用的局部随机种子
    this_Seed = randi(2^31 - 1);

    % 5) 用高精度字符串作为唯一键，检查这组参数是否已经出现过
    key = make_param_key(this_Hs, this_Tz, this_Dir);
    if isKey(used_keys, key)
        continue;
    end

    % 6) 如果没重复，就收进参数池
    used_keys(key) = true;
    param_pool(sample_idx) = struct( ...
        'Hs', this_Hs, ...
        'Tz', this_Tz, ...
        'Dir', this_Dir, ...
        'Seed', this_Seed ...
    );

    if mod(sample_idx, 200) == 0 || sample_idx == total_samples
        fprintf('唯一参数池进度: %d / %d\n', sample_idx, total_samples);
    end

    sample_idx = sample_idx + 1;
end

% 生成完参数池后统一打乱，再切分到 train / val / test。
% 这样可以避免“前一段参数都进 train，后一段参数都进 test”的顺序偏置。
perm = randperm(total_samples);
param_pool = param_pool(perm);

%% 5. 在划分固定之后，再逐个生成 tensor 并保存
% 到这里为止，split 已经确定好了。
% 后面的 Generate_Clip 只负责根据参数生成样本，
% 不再影响 train / val / test 的独立性。
global_counter = 0;
start_time = tic;
pool_start = 1;

for k = 1:size(tasks, 1)
    num_samples = tasks{k, 1};
    save_path   = tasks{k, 2};
    task_name   = tasks{k, 3};

    if num_samples == 0
        continue;
    end

    % 从总参数池中切出当前 split 对应的参数段
    pool_end = pool_start + num_samples - 1;
    split_pool = param_pool(pool_start:pool_end);
    pool_start = pool_end + 1;

    fprintf('\n=== 开始生成 %s 集 (%d 个样本) ===\n', task_name, num_samples);

    for i = 1:num_samples
        sample = split_pool(i);

        % 根据当前样本的参数与局部种子，生成点云序列与标签
        [tensor_data, labels] = Generate_Clip( ...
            sample.Hs, sample.Tz, sample.Dir, sample.Seed, clip_settings);

        data_to_save = struct();
        data_to_save.tensor = tensor_data;
        data_to_save.labels = labels;
        data_to_save.params = [sample.Hs, sample.Tz, sample.Dir];
        data_to_save.seed = sample.Seed;

        % 文件名中只保留简化后的显示数值。
        % 真正的精确参数仍保存在 .mat 文件内部的 params 中。
        fname = sprintf('Sample_%04d_Hs%.1f_Tz%.1f_Dirdeg%.0f.mat', ...
            i, sample.Hs, sample.Tz, rad2deg(sample.Dir));
        full_path = fullfile(save_path, fname);
        save(full_path, '-struct', 'data_to_save');

        global_counter = global_counter + 1;

        if mod(i, 10) == 0 || i == num_samples
            elapsed = toc(start_time);
            avg_time = elapsed / max(global_counter, 1);
            remain_time = avg_time * (total_samples - global_counter);

            fprintf('[%s] 进度: %d/%d | 已耗时: %.1fs | 平均每个样本: %.2fs | 剩余: %.0fs\n', ...
                task_name, i, num_samples, elapsed, avg_time, remain_time);
        end
    end
end

fprintf('\n所有数据已生成完成，保存目录: %s\n', base_dir);

function key = make_param_key(Hs, Tz, Dir)
% 用较高精度把参数转成字符串键，用于判重。
% 这里只对参数组合判重，不对 seed 判重。
key = sprintf('%.10f_%.10f_%.10f', Hs, Tz, Dir);
end
