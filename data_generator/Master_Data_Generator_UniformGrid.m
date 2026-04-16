clc; clear; close all;

% Create a second dataset generation pipeline that keeps the original wave
% spectrum / parameter sampling logic unchanged, but samples the wave field
% on a spatially uniform 101x101 X-Y grid instead of an equal-angle scan.

%% 1. Check GPU
if gpuDeviceCount("available") < 1
    error('No CUDA-capable GPU is available.');
else
    g = gpuDevice(1);
    fprintf('=== Using GPU ===\n');
    fprintf('Device: %s\n', g.Name);
    fprintf('Free memory: %.2f GB\n', g.AvailableMemory / 1e9);
    fprintf('=================\n');
    reset(g);
end

%% 2. Output setup
base_dir = '../datasets/Dataset_Wave_Lidar_10000samples_uniform_grid';
n_train = 8000;
n_val   = 1000;
n_test  = 1000;

% Keep the same master seed and clip settings as the original generator.
master_seed = 20260412;

clip_settings.duration = 16;
clip_settings.fps = 4;
clip_settings.grid_size = 101;
clip_settings.roi = 20;

range.Hs = [1.0, 5.5];
range.Tz = [4.0, 8.0];
range.Dir = [0, 2*pi];

%% 3. Create folders
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

tasks = {
    n_train, folders{1}, 'Train'; ...
    n_val,   folders{2}, 'Val'; ...
    n_test,  folders{3}, 'Test'
};

%% 4. Build one shared parameter pool, then split train/val/test
total_samples = n_train + n_val + n_test;
rng(master_seed, 'twister');

param_pool(total_samples) = struct('Hs', 0, 'Tz', 0, 'Dir', 0, 'Seed', 0);
used_keys = containers.Map('KeyType', 'char', 'ValueType', 'logical');

sample_idx = 1;
attempt_count = 0;
max_attempts = total_samples * 500;

fprintf('\nGenerating %d unique parameter sets...\n', total_samples);

while sample_idx <= total_samples
    attempt_count = attempt_count + 1;
    if attempt_count > max_attempts
        error('Failed to sample enough unique wave parameter combinations.');
    end

    this_Hs = range.Hs(1) + rand() * (range.Hs(2) - range.Hs(1));

    min_T = sqrt(this_Hs) * 2.5;
    actual_min_T = max(min_T, range.Tz(1));
    this_Tz = actual_min_T + rand() * (range.Tz(2) - actual_min_T);

    this_Dir = range.Dir(1) + rand() * (range.Dir(2) - range.Dir(1));
    this_Seed = randi(2^31 - 1);

    key = make_param_key(this_Hs, this_Tz, this_Dir);
    if isKey(used_keys, key)
        continue;
    end

    used_keys(key) = true;
    param_pool(sample_idx) = struct( ...
        'Hs', this_Hs, ...
        'Tz', this_Tz, ...
        'Dir', this_Dir, ...
        'Seed', this_Seed ...
    );

    if mod(sample_idx, 200) == 0 || sample_idx == total_samples
        fprintf('Parameter sets prepared: %d / %d\n', sample_idx, total_samples);
    end

    sample_idx = sample_idx + 1;
end

perm = randperm(total_samples);
param_pool = param_pool(perm);

%% 5. Generate and save clips
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

    pool_end = pool_start + num_samples - 1;
    split_pool = param_pool(pool_start:pool_end);
    pool_start = pool_end + 1;

    fprintf('\n=== Generating %s split (%d samples) ===\n', task_name, num_samples);

    for i = 1:num_samples
        sample = split_pool(i);

        [tensor_data, labels, grid_meta] = Generate_Clip_UniformGrid( ...
            sample.Hs, sample.Tz, sample.Dir, sample.Seed, clip_settings);

        data_to_save = struct();
        data_to_save.tensor = tensor_data;
        data_to_save.labels = labels;
        data_to_save.params = [sample.Hs, sample.Tz, sample.Dir];
        data_to_save.seed = sample.Seed;
        data_to_save.grid_meta = grid_meta;

        fname = sprintf('Sample_%04d_Hs%.1f_Tz%.1f_Dirdeg%.0f.mat', ...
            i, sample.Hs, sample.Tz, rad2deg(sample.Dir));
        full_path = fullfile(save_path, fname);
        save(full_path, '-struct', 'data_to_save');

        global_counter = global_counter + 1;

        if mod(i, 10) == 0 || i == num_samples
            elapsed = toc(start_time);
            avg_time = elapsed / max(global_counter, 1);
            remain_time = avg_time * (total_samples - global_counter);

            fprintf('[%s] Progress: %d/%d | Elapsed: %.1fs | Avg/sample: %.2fs | ETA: %.0fs\n', ...
                task_name, i, num_samples, elapsed, avg_time, remain_time);
        end
    end
end

fprintf('\nUniform-grid dataset generation finished. Output: %s\n', base_dir);

function key = make_param_key(Hs, Tz, Dir)
key = sprintf('%.10f_%.10f_%.10f', Hs, Tz, Dir);
end
