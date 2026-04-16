# lidar-wave-estimate

Wave height (`Hs`) and wave direction estimation from LiDAR wave-surface data using CNN / ConvLSTM models.

## Overview

This project trains a neural network on simulated `101 x 101` LiDAR wave-surface grids and predicts:

- wave direction as a 2D unit vector: `(sin(dir), cos(dir))`
- wave height: `Hs`

The current default setup uses:

- temporal length: `64` frames
- spatial size: `101 x 101`
- model: `ConvLSTM`
- input representation:
  - raw height grid `z`
  - model-internal coordinate priors `rr` and `theta`

The model input tensor shape is:

```python
[B, 1, T, H, W]
```

The extra two channels are generated inside the model, so the external data pipeline still provides one raw grid channel per frame.

## Repository Structure

```text
configs/
  config.py                 # training and inference config
data/
  dataset.py                # .mat dataset loader and augmentation
models/
  wave_cnn.py               # FrameEncoder + ConvLSTM / PureCNN models
train.py                    # model training
evaluate.py                 # evaluation and plotting
predict_real_data.py        # helper for converting real point clouds to model input
project_xyz_frame.py        # project one .xyz frame to the model grid and visualize it
visualize_augmentation.py   # inspect augmentation effects
Generate_Clip.m             # synthetic sample generation
Master_Data_Generator.m     # dataset generation entry
Visualize_One_Frame.m       # MATLAB visualization
```

## Data Format

### Synthetic training data

The training / validation / test sets are stored as `.mat` files under:

```text
<data_root>/
  train/
  val/
  test/
```

Each `.mat` sample is expected to contain:

- `tensor`: shape `[T, H, W]`
- `labels`: shape `[3]`, formatted as:
  - `labels[0] = sin(dir)`
  - `labels[1] = cos(dir)`
  - `labels[2] = Hs`

### Real point cloud data

Real data can be converted from irregular `.xyz` point clouds into the grid format expected by the model.

- one frame: use `project_xyz_frame.py`
- multiple frames: use `predict_real_data.py` as a starting point and keep the same grid definition across all frames

## Configuration

Main settings are in `configs/config.py`.

Current defaults:

```python
frames = 64
height = 101
width = 101
lidar_scale = 5.0
model_name = "ConvLSTM"
temporal_pool = "max"
temporal_stride = 2
```

Adjust these before training if your experiment setup changes.

## Training

Run:

```bash
python train.py
```

Training outputs are written to:

- `checkpoints/...`
- `logs/...`
- `results/...`

The best checkpoint is saved as:

```text
best_model.pth
```

## Evaluation

Run:

```bash
python evaluate.py
```

Choose a specific experiment folder:

```bash
python evaluate.py \
  --model-name TemporalTransformer \
  --experiment-tag datasetsv3_realdataloss_80drop
```

Choose a specific checkpoint inside one experiment:

```bash
python evaluate.py \
  --experiment-name ConvLSTM_dataloss_datasetsv2 \
  --checkpoint-file epoch_40.pth
```

Choose a checkpoint directly, and outputs will default to that experiment's `results` directory:

```bash
python evaluate.py \
  --checkpoint-path ./all_exps_result/ConvLSTM_dataloss_datasetsv2/checkpoints/best_model.pth
```

This script:

- loads `best_model.pth`
- evaluates on the test split
- writes detailed CSV results
- plots regression/error figures

## Real `.xyz` Frame Projection

To project one `.xyz` frame into the `101 x 101` model grid and compare raw vs projected geometry:

```bash
python project_xyz_frame.py \
  --xyz path/to/frame.xyz \
  --save-fig results/real_frame_projection.png \
  --save-npz results/real_frame_projection.npz
```

Windows PowerShell example:

```powershell
python .\project_xyz_frame.py `
  --xyz .\your_frame.xyz `
  --save-fig .\results\real_frame_projection.png `
  --save-npz .\results\real_frame_projection.npz
```

The script outputs:

- raw point cloud plot
- projected grid point cloud plot
- projected 2D height map

Useful options:

- `--roi`: symmetric XY range used for projection
- `--x-min --x-max --y-min --y-max`: manual projection bounds
- `--agg`: aggregation method for multiple points in one cell
- `--placement`: how a single frame is inserted into a `T x H x W` tensor

## Notes on Real Inference

The current model was trained on simulated, regularized grids. Real point clouds must be preprocessed to match the training representation:

1. convert each frame to a fixed `101 x 101` grid
2. keep all frames in the same coordinate system
3. normalize using the same `lidar_scale`
4. stack frames into `[1, 1, T, H, W]`

Important:

- `z` comes from the projected height grid
- `rr` and `theta` are generated from the grid coordinates inside the model
- if grid size and ROI stay fixed, `rr` and `theta` do not need to be recomputed manually per frame

## Environment

Recommended Python packages:

```text
torch
numpy
scipy
matplotlib
seaborn
pandas
scikit-learn
tqdm
```

MATLAB is required only for synthetic data generation scripts.

## Current Status

This repository is under active iteration. The current codebase includes:

- ConvLSTM and PureCNN variants
- synthetic-data training pipeline
- augmentation visualization
- single-frame `.xyz` projection tool
- local Git history initialization for experiment tracking
