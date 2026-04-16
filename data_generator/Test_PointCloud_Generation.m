clc; clear; close all;

% Quick visualization script for comparing:
% 1) the original equal-angle LiDAR scan sampling
% 2) the new uniform 101x101 spatial grid sampling
%
% It uses the same wave parameters for both generators so the spatial
% distribution difference is easy to see.

%% Example wave parameters
Hs = 3.0;
Tz = 6.0;
MainDir = deg2rad(45);
Seed = 20260412;

%% Shared settings
settings.duration = 16;
settings.fps = 4;
settings.grid_size = 101;
settings.roi = 20;

%% Generate one clip from each pipeline
[range_tensor, labels_radar] = Generate_Clip(Hs, Tz, MainDir, Seed, settings);
[height_tensor, labels_uniform, grid_meta] = Generate_Clip_UniformGrid(Hs, Tz, MainDir, Seed, settings);

frames = size(range_tensor, 1);
frame_idx = ceil(frames / 2);

%% Recover XYZ point cloud from the original radar-style range delta
[X_radar, Y_radar, Z_radar, Delta_range] = reconstruct_radar_frame_xyz(range_tensor(frame_idx, :, :));

%% Build XYZ point cloud for the uniform grid sampling
[X_uniform, Y_uniform] = meshgrid(grid_meta.x_coords, grid_meta.y_coords);
Z_uniform = squeeze(height_tensor(frame_idx, :, :));

%% Basic statistics
fprintf('Original radar-style tensor size: [%d, %d, %d]\n', size(range_tensor, 1), size(range_tensor, 2), size(range_tensor, 3));
fprintf('Uniform grid tensor size: [%d, %d, %d]\n', size(height_tensor, 1), size(height_tensor, 2), size(height_tensor, 3));
fprintf('Frame visualized: %d / %d\n', frame_idx, frames);
fprintf('Labels (radar):   [sin(dir), cos(dir), Hs] = [%.4f, %.4f, %.4f]\n', labels_radar(1), labels_radar(2), labels_radar(3));
fprintf('Labels (uniform): [sin(dir), cos(dir), Hs] = [%.4f, %.4f, %.4f]\n', labels_uniform(1), labels_uniform(2), labels_uniform(3));

%% Plot
fig = figure('Color', 'w', 'Name', 'Point Cloud Sampling Comparison');
tiledlayout(fig, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
scatter3(X_radar(:), Y_radar(:), Z_radar(:), 10, Z_radar(:), 'filled');
title(sprintf('Original Radar Sampling (Frame %d)', frame_idx));
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
axis equal;
grid on;
view(35, 25);
colorbar;

nexttile;
scatter(X_radar(:), Y_radar(:), 10, Z_radar(:), 'filled');
title('Original Radar Sampling Top View');
xlabel('X (m)');
ylabel('Y (m)');
axis equal;
grid on;
colorbar;

nexttile;
scatter3(X_uniform(:), Y_uniform(:), Z_uniform(:), 10, Z_uniform(:), 'filled');
title(sprintf('Uniform 101x101 Sampling (Frame %d)', frame_idx));
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
axis equal;
grid on;
view(35, 25);
colorbar;

nexttile;
surf(X_uniform, Y_uniform, Z_uniform, 'EdgeColor', 'none');
title('Uniform 101x101 Surface');
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
axis equal;
grid on;
view(35, 25);
colorbar;

sgtitle(sprintf('Hs = %.2f m, Tz = %.2f s, Dir = %.1f deg, Seed = %d', ...
    Hs, Tz, rad2deg(MainDir), Seed));

%% Optional: show the radar delta-range frame itself
figure('Color', 'w', 'Name', 'Original Delta Range Frame');
imagesc(Delta_range);
axis image;
title(sprintf('Original Delta Range (Frame %d)', frame_idx));
xlabel('Column');
ylabel('Row');
colorbar;

function [X, Y, Z, DeltaRange] = reconstruct_radar_frame_xyz(frame_2d)
    lidar_height = 15;
    fov = 68;
    res = 0.68;

    DeltaRange = squeeze(frame_2d);

    angs = -fov / 2 : res : fov / 2;
    [AlphaGrid, BetaGrid] = meshgrid(deg2rad(angs), deg2rad(angs));

    Sx = tan(AlphaGrid);
    Sy = tan(BetaGrid) ./ cos(AlphaGrid);
    ray_norm = sqrt(1 + Sx.^2 + Sy.^2);

    dist_calm = lidar_height * ray_norm;
    dist_wave = dist_calm - DeltaRange;

    Z = lidar_height - dist_wave ./ ray_norm;
    X = (dist_wave ./ ray_norm) .* Sx;
    Y = (dist_wave ./ ray_norm) .* Sy;
end
