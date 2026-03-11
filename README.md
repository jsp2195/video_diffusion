# Lightweight First-Frame Conditioned Grayscale Video Diffusion (PyTorch)

This repository now defaults to a **compact pixel-space 3D U-Net DDPM baseline** for:

- `p(video | first_frame)`
- grayscale MP4s
- fast, stable iteration on limited hardware

## Default model path (active)

- Compact 3D U-Net denoiser (`models/video_unet3d.py`)
- Input noisy clip shape: `[B,1,T,H,W]`
- Input condition frame shape: `[B,1,H,W]`
- Conditioning method: repeat condition over time and concatenate (`[B,2,T,H,W]`)
- v-prediction objective
- DDPM training + DDIM sampling
- EMA for previews/sampling
- AMP enabled by default (`--amp`, disable with `--no-amp`)
- Gradient clipping
- Classifier-free condition dropout
- Optional tiny temporal smoothness loss (`--temporal_loss_weight`, default `0.02`)

### Compact defaults

- `size=64`
- `T=8`
- `base_channels=64`
- `channel_mults=1 2 4`
- `res_blocks=2`

## What changed from heavier paths

- The default workflow no longer uses transformer-style denoisers.
- The default workflow is no longer configured around heavy high-resolution / long-clip settings.
- The active train and sample scripts instantiate the compact 3D U-Net by default.

Legacy modules may still exist in the repo for reference, but they are not wired as the primary path.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install decord imageio[ffmpeg] numpy pillow pyyaml
```

## Dataset layout

Expected structure:

```text
<data_root>/
  videos_val/
    .../*.mp4
```

Loader contract:

- `cond`: `[1,H,W]` (first frame)
- `clip`: `[1,T,H,W]` (grayscale clip in `[-1,1]`)

## First overfit test (recommended first run)

```bash
python train_video_ddpm.py \
  --data_root /path/to/data_root \
  --out_dir ./outputs/overfit_test \
  --max_videos 64 \
  --batch_size 2 \
  --epochs 20 \
  --max_steps 500 \
  --vis_every 1 \
  --num_workers 2
```

## Normal lightweight training run

```bash
python train_video_ddpm.py \
  --data_root /path/to/data_root \
  --out_dir ./outputs/train_light \
  --size 64 \
  --T 8 \
  --batch_size 8 \
  --epochs 30 \
  --num_workers 4 \
  --lr 1e-4 \
  --vis_every 1 \
  --cfg_drop_prob 0.15 \
  --amp
```

## Sampling (EMA + DDIM)

```bash
python sample_video_ddpm.py \
  --ckpt ./outputs/train_light/last.pt \
  --data_root /path/to/data_root \
  --out_dir ./outputs/sample \
  --size 64 \
  --T 8 \
  --ddim_steps 40 \
  --cfg_scale 1.8
```

or with a custom condition image:

```bash
python sample_video_ddpm.py \
  --ckpt ./outputs/train_light/last.pt \
  --cond_image ./cond.png \
  --out_dir ./outputs/sample_custom \
  --size 64 \
  --T 8
```

Outputs:

- `cond.png`
- `sample.mp4`

## Key CLI options

`train_video_ddpm.py`:
- `--data_root`
- `--out_dir`
- `--size`
- `--T`
- `--batch_size`
- `--epochs`
- `--num_workers`
- `--lr`
- `--amp/--no-amp`
- `--vis_every`
- `--cfg_drop_prob`

## Current limitations

- This is a lightweight baseline for quick success, not SOTA quality.
- Motion quality depends strongly on dataset diversity and clip count.
- For best stability, first overfit a tiny subset, then scale data.
