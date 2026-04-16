function [HeightTensor, LabelVector, GridMeta] = Generate_Clip_UniformGrid(Hs, Tz, MainDir, Seed, settings)
    % Generate one clip using the same wave-field synthesis as Generate_Clip,
    % but sample the surface on a spatially uniform X-Y grid instead of an
    % equal-angle LiDAR scan grid.
    %
    % Output:
    %   HeightTensor: [frames, grid_size, grid_size], surface height z(x, y, t)
    %   LabelVector : [sin(MainDir), cos(MainDir), Hs]
    %   GridMeta    : optional metadata for the uniform spatial grid

    frames = round(settings.duration * settings.fps);
    dt = 1 / settings.fps;
    grid_size = settings.grid_size;
    roi = settings.roi;

    local_stream = RandStream('mt19937ar', 'Seed', double(Seed));

    % Keep the original wave spectrum construction unchanged.
    g = 9.81;
    n_freq = 300;
    n_dir = 72;
    s_max = 25;

    wp = 2 * pi / (Tz * 1.414);
    w_min = 0.5 * wp;
    w_max = 8.0 * wp;

    w_edges = linspace(w_min, w_max, n_freq + 1)';
    w_noise = rand(local_stream, n_freq, 1);
    w = w_edges(1:end-1) + w_noise .* diff(w_edges);
    dw = gradient(w);

    theta_edges = linspace(-pi / 2, pi / 2, n_dir + 1);
    theta_noise = rand(local_stream, 1, n_dir);
    theta_local = theta_edges(1:end-1) + theta_noise .* diff(theta_edges);
    dtheta = gradient(theta_local);

    S_w = (173 * Hs^2 / Tz^4) ./ (w.^5) .* exp(-691 ./ (Tz^4 .* w.^4));

    S_spread_vec = zeros(n_freq, 1);
    for k = 1:n_freq
        if w(k) <= wp
            S_spread_vec(k) = s_max * (w(k) / wp)^5;
        else
            S_spread_vec(k) = s_max * (w(k) / wp)^(-2.5);
        end
    end

    [W_grid, Theta_local_grid] = meshgrid(w, theta_local);

    S_param_grid = repmat(S_spread_vec', n_dir, 1);
    D_raw = max(0, cos(Theta_local_grid)).^(2 * S_param_grid);

    dtheta_col = dtheta(:);
    dtheta_grid = repmat(dtheta_col, 1, n_freq);

    Integral_D = sum(D_raw .* dtheta_grid, 1);
    Norm_Factor = 1 ./ Integral_D;
    Norm_Grid = repmat(Norm_Factor, n_dir, 1);
    D_grid = D_raw .* Norm_Grid;

    S_w_grid = repmat(S_w', n_dir, 1);
    Spec_2D = S_w_grid .* D_grid;

    dw_col = dw(:);
    dw_grid = repmat(dw_col', n_dir, 1);
    Amp = sqrt(2 * Spec_2D .* dw_grid .* dtheta_grid);

    m0_sim = sum(Amp(:).^2) / 2;
    Hs_sim_raw = 4 * sqrt(m0_sim);
    if Hs_sim_raw > 0
        Amp = Amp * (Hs / Hs_sim_raw);
    end

    Phases = rand(local_stream, n_dir, n_freq) * 2 * pi;

    Theta_global = Theta_local_grid + MainDir;
    K_num = (W_grid.^2) / g;
    Kx = K_num .* cos(Theta_global);
    Ky = K_num .* sin(Theta_global);

    p_amp = gpuArray(single(Amp(:)));
    p_kx = gpuArray(single(Kx(:)));
    p_ky = gpuArray(single(Ky(:)));
    p_w = gpuArray(single(W_grid(:)));
    p_phi = gpuArray(single(Phases(:)));

    % Uniform spatial sampling grid in X-Y.
    x_vec = linspace(-roi, roi, grid_size);
    y_vec = linspace(-roi, roi, grid_size);
    [Query_X, Query_Y] = meshgrid(x_vec, y_vec);

    Query_X = gpuArray(single(Query_X));
    Query_Y = gpuArray(single(Query_Y));
    flat_Query_X = Query_X(:)';
    flat_Query_Y = Query_Y(:)';

    HeightTensor_GPU = zeros(frames, grid_size, grid_size, 'single', 'gpuArray');

    block_size = 5000;
    num_components = length(p_amp);

    for t = 1:frames
        cur_time = (t - 1) * dt;
        Z_flat = zeros(1, length(flat_Query_X), 'single', 'gpuArray');

        for k = 1:block_size:num_components
            idx_end = min(k + block_size - 1, num_components);
            idx_range = k:idx_end;

            spatial_phase = p_kx(idx_range) * flat_Query_X + p_ky(idx_range) * flat_Query_Y;
            time_phase = p_phi(idx_range) - p_w(idx_range) * cur_time;
            Z_flat = Z_flat + sum(p_amp(idx_range) .* cos(spatial_phase + time_phase), 1);
        end

        HeightTensor_GPU(t, :, :) = reshape(Z_flat, size(Query_X));
    end

    HeightTensor = gather(HeightTensor_GPU);
    LabelVector = [sin(MainDir), cos(MainDir), Hs];

    if nargout > 2
        GridMeta = struct( ...
            'x_coords', single(x_vec), ...
            'y_coords', single(y_vec), ...
            'roi', single(roi), ...
            'grid_size', int32(grid_size), ...
            'sampling_mode', 'uniform_xy', ...
            'value_type', 'surface_height');
    end
end
