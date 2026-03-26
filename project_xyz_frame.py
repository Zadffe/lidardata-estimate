import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from configs.config import Config


def load_xyz_frame(xyz_path):
    loaders = (
        lambda path: np.loadtxt(path, dtype=np.float32),
        lambda path: np.loadtxt(path, dtype=np.float32, delimiter=","),
    )

    last_error = None
    for loader in loaders:
        try:
            points = loader(xyz_path)
            break
        except Exception as exc:  # pragma: no cover
            last_error = exc
    else:
        raise ValueError(f"Failed to load xyz file: {xyz_path}\n{last_error}")

    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in {xyz_path}, got shape {points.shape}")

    points = points[:, :3]
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if len(points) == 0:
        raise ValueError(f"No valid xyz points found in {xyz_path}")
    return points


def resolve_ranges(points, roi, x_range, y_range):
    if x_range is None:
        if roi is not None:
            half = float(roi) / 2.0
            x_range = (-half, half)
        else:
            max_abs_x = float(np.max(np.abs(points[:, 0])))
            x_range = (-max_abs_x, max_abs_x)

    if y_range is None:
        if roi is not None:
            half = float(roi) / 2.0
            y_range = (-half, half)
        else:
            max_abs_y = float(np.max(np.abs(points[:, 1])))
            y_range = (-max_abs_y, max_abs_y)

    if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
        raise ValueError("Invalid x/y range: min must be smaller than max")

    return tuple(map(float, x_range)), tuple(map(float, y_range))


def project_points_to_grid(points, height, width, x_range, y_range, agg="mean", fill_value=0.0):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    x_min, x_max = x_range
    y_min, y_max = y_range

    grid_x = np.floor((x - x_min) / (x_max - x_min) * width).astype(np.int32)
    grid_y = np.floor((y - y_min) / (y_max - y_min) * height).astype(np.int32)

    valid = (
        (grid_x >= 0)
        & (grid_x < width)
        & (grid_y >= 0)
        & (grid_y < height)
        & np.isfinite(z)
    )

    grid_x = grid_x[valid]
    grid_y = grid_y[valid]
    z = z[valid]

    grid = np.full((height, width), fill_value, dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=bool)

    if len(z) == 0:
        return grid, valid_mask

    if agg == "mean":
        sum_grid = np.zeros((height, width), dtype=np.float64)
        cnt_grid = np.zeros((height, width), dtype=np.int32)
        np.add.at(sum_grid, (grid_y, grid_x), z)
        np.add.at(cnt_grid, (grid_y, grid_x), 1)
        valid_mask = cnt_grid > 0
        grid[valid_mask] = (sum_grid[valid_mask] / cnt_grid[valid_mask]).astype(np.float32)
    elif agg == "max":
        grid[:] = -np.inf
        np.maximum.at(grid, (grid_y, grid_x), z)
        valid_mask = np.isfinite(grid)
        grid[~valid_mask] = fill_value
    elif agg == "min":
        grid[:] = np.inf
        np.minimum.at(grid, (grid_y, grid_x), z)
        valid_mask = np.isfinite(grid)
        grid[~valid_mask] = fill_value
    elif agg == "last":
        grid[grid_y, grid_x] = z
        valid_mask[grid_y, grid_x] = True
    else:
        raise ValueError("agg must be one of: mean, max, min, last")

    return grid, valid_mask


def grid_to_projected_points(grid, valid_mask, x_range, y_range):
    height, width = grid.shape
    ys, xs = np.where(valid_mask)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    x_centers = np.linspace(x_range[0], x_range[1], width, dtype=np.float32)
    y_centers = np.linspace(y_range[0], y_range[1], height, dtype=np.float32)

    proj_points = np.column_stack(
        [
            x_centers[xs],
            y_centers[ys],
            grid[ys, xs],
        ]
    ).astype(np.float32)
    return proj_points


def build_model_input(frame_grid, frames, lidar_scale, placement="center"):
    volume = np.zeros((frames, frame_grid.shape[0], frame_grid.shape[1]), dtype=np.float32)
    frame_norm = (frame_grid / float(lidar_scale)).astype(np.float32)

    if placement == "repeat":
        volume[:] = frame_norm[None, :, :]
    elif placement == "center":
        volume[frames // 2] = frame_norm
    elif placement == "first":
        volume[0] = frame_norm
    else:
        raise ValueError("placement must be one of: center, first, repeat")

    return volume[None, None, :, :, :], frame_norm


def _set_equal_axes(ax, points):
    if len(points) == 0:
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_comparison(raw_points, projected_points, frame_grid, valid_mask, out_path=None, title=None):
    fig = plt.figure(figsize=(18, 6))

    ax_raw = fig.add_subplot(1, 3, 1, projection="3d")
    sc_raw = ax_raw.scatter(
        raw_points[:, 0],
        raw_points[:, 1],
        raw_points[:, 2],
        c=raw_points[:, 2],
        s=2,
        cmap="viridis",
        alpha=0.7,
        depthshade=False,
    )
    _set_equal_axes(ax_raw, raw_points)
    ax_raw.set_title(f"Raw XYZ\npoints={len(raw_points)}")
    ax_raw.set_xlabel("X")
    ax_raw.set_ylabel("Y")
    ax_raw.set_zlabel("Z")
    fig.colorbar(sc_raw, ax=ax_raw, shrink=0.7, pad=0.05, label="Z")

    ax_proj = fig.add_subplot(1, 3, 2, projection="3d")
    if len(projected_points) > 0:
        sc_proj = ax_proj.scatter(
            projected_points[:, 0],
            projected_points[:, 1],
            projected_points[:, 2],
            c=projected_points[:, 2],
            s=8,
            cmap="viridis",
            alpha=0.9,
            depthshade=False,
        )
        _set_equal_axes(ax_proj, projected_points)
        fig.colorbar(sc_proj, ax=ax_proj, shrink=0.7, pad=0.05, label="Z")
    ax_proj.set_title(f"Projected Grid Cloud\nvalid_cells={int(valid_mask.sum())}")
    ax_proj.set_xlabel("X")
    ax_proj.set_ylabel("Y")
    ax_proj.set_zlabel("Z")

    ax_map = fig.add_subplot(1, 3, 3)
    display_grid = np.ma.masked_where(~valid_mask, frame_grid)
    im = ax_map.imshow(display_grid, origin="lower", cmap="viridis")
    ax_map.set_title("Projected Height Map")
    ax_map.set_xlabel("Grid X")
    ax_map.set_ylabel("Grid Y")
    fig.colorbar(im, ax=ax_map, shrink=0.8, pad=0.02, label="Z")

    if title:
        fig.suptitle(title)
    plt.tight_layout()

    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"Saved figure to: {out_path}")

    plt.show()


def main():
    cfg = Config()

    parser = argparse.ArgumentParser(
        description="Project one xyz frame to the model grid and visualize raw vs projected point clouds."
    )
    parser.add_argument("--xyz", required=True, help="Path to one .xyz frame file.")
    parser.add_argument("--height", type=int, default=cfg.height, help="Grid height. Default: config height.")
    parser.add_argument("--width", type=int, default=cfg.width, help="Grid width. Default: config width.")
    parser.add_argument(
        "--frames",
        type=int,
        default=cfg.frames,
        help="Temporal length for exported model input volume. Default: config frames.",
    )
    parser.add_argument(
        "--roi",
        type=float,
        default=20.0,
        help="Symmetric XY ROI size in meters. Used if x/y ranges are not set. Default: 20.0.",
    )
    parser.add_argument("--x-min", type=float, default=None, help="Optional manual X min.")
    parser.add_argument("--x-max", type=float, default=None, help="Optional manual X max.")
    parser.add_argument("--y-min", type=float, default=None, help="Optional manual Y min.")
    parser.add_argument("--y-max", type=float, default=None, help="Optional manual Y max.")
    parser.add_argument(
        "--agg",
        type=str,
        default="mean",
        choices=["mean", "max", "min", "last"],
        help="How to merge multiple points falling into the same cell.",
    )
    parser.add_argument(
        "--placement",
        type=str,
        default="center",
        choices=["center", "first", "repeat"],
        help="How to place the projected frame inside the exported T x H x W model volume.",
    )
    parser.add_argument(
        "--save-fig",
        type=str,
        default=None,
        help="Optional output image path for the comparison figure.",
    )
    parser.add_argument(
        "--save-npz",
        type=str,
        default=None,
        help="Optional output .npz path containing projected frame and model-ready volume.",
    )
    args = parser.parse_args()

    xyz_path = os.path.abspath(args.xyz)
    points = load_xyz_frame(xyz_path)

    x_range_input = None if args.x_min is None or args.x_max is None else (args.x_min, args.x_max)
    y_range_input = None if args.y_min is None or args.y_max is None else (args.y_min, args.y_max)
    x_range, y_range = resolve_ranges(points, args.roi, x_range_input, y_range_input)

    frame_grid, valid_mask = project_points_to_grid(
        points=points,
        height=args.height,
        width=args.width,
        x_range=x_range,
        y_range=y_range,
        agg=args.agg,
        fill_value=0.0,
    )
    projected_points = grid_to_projected_points(frame_grid, valid_mask, x_range, y_range)
    model_input, frame_norm = build_model_input(
        frame_grid=frame_grid,
        frames=args.frames,
        lidar_scale=cfg.lidar_scale,
        placement=args.placement,
    )

    print(f"Loaded raw points       : {len(points)}")
    print(f"Projected valid cells   : {int(valid_mask.sum())}")
    print(f"Frame grid shape        : {frame_grid.shape}")
    print(f"Model input shape       : {model_input.shape}")
    print(f"X range                 : [{x_range[0]:.3f}, {x_range[1]:.3f}]")
    print(f"Y range                 : [{y_range[0]:.3f}, {y_range[1]:.3f}]")
    print(f"Aggregation             : {args.agg}")
    print(f"Placement               : {args.placement}")

    if args.save_npz:
        save_npz_path = os.path.abspath(args.save_npz)
        save_dir = os.path.dirname(save_npz_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(
            save_npz_path,
            raw_points=points,
            projected_frame=frame_grid.astype(np.float32),
            projected_frame_norm=frame_norm.astype(np.float32),
            valid_mask=valid_mask.astype(np.uint8),
            model_input=model_input.astype(np.float32),
            x_range=np.asarray(x_range, dtype=np.float32),
            y_range=np.asarray(y_range, dtype=np.float32),
        )
        print(f"Saved projected outputs : {save_npz_path}")

    title = (
        f"{os.path.basename(xyz_path)} | "
        f"grid={args.height}x{args.width} | "
        f"model_input={model_input.shape}"
    )
    plot_comparison(
        raw_points=points,
        projected_points=projected_points,
        frame_grid=frame_grid,
        valid_mask=valid_mask,
        out_path=args.save_fig,
        title=title,
    )


if __name__ == "__main__":
    main()
