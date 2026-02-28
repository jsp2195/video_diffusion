# Grayscale Conditional Video DDPM (PyTorch MVP)

This repo refactors the single-image conditional DDPM baseline into a **first-frame conditioned grayscale video DDPM**.

## Core tensor contract

- `cond_first_frame`: `[B,1,H,W]`
- `clip`: `[B,1,T,H,W]`
- `pred_noise = model(noisy_clip, t, cond_first_frame)`
- `noisy_clip`: `[B,1,T,H,W]`
- `t`: `[B]` (`int64`)
- `pred_noise`: `[B,1,T,H,W]`

The training objective is unchanged: epsilon-prediction MSE (`MSE(pred_noise, noise)`).

## Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install decord imageio[ffmpeg] numpy pillow
```

## Dataset setup (optional downloader integration)

If mp4 files already exist under `data_root/videos_val/**/*.mp4`, you can skip downloader setup entirely.

### 1) Unzip downloader package

```bash
bash scripts/setup_kinetics_downloader.sh /mnt/data/kinetics-dataset-main.zip vendor/kinetics_downloader
```

### 2) Run downloader scripts (if needed)

After unzip, inspect available scripts:

```bash
find vendor/kinetics_downloader -maxdepth 3 -type f \( -name '*.sh' -o -name 'k400_*' \)
```

Typical usage (depends on downloaded package contents):

```bash
bash vendor/kinetics_downloader/kinetics-dataset-main/download.sh
bash vendor/kinetics_downloader/kinetics-dataset-main/extract.sh
```

Expected layout used by this training code:

```text
data_root/
  videos_val/
    .../*.mp4
```

The loader discovers videos using recursive glob on `data_root/videos_val/**/*.mp4`.

## Smoke test

```bash
bash scripts/smoke_subset.sh /path/to/data_root ./outputs/smoke
```

Equivalent direct command:

```bash
python train_video_ddpm.py \
  --data_root /path/to/data_root \
  --val_ratio 0.01 \
  --max_videos 200 \
  --max_steps 30 \
  --shape_check
```

## Train

```bash
python train_video_ddpm.py \
  --data_root /path/to/data_root \
  --out_dir ./outputs/train \
  --val_ratio 0.01 \
  --T 16 \
  --size 128 \
  --batch_size 2 \
  --lr 1e-4 \
  --epochs 20 \
  --timesteps 1000 \
  --cfg_drop_prob 0.1 \
  --ema_decay 0.999 \
  --save_every 500 \
  --num_workers 4 \
  --seed 42 \
  --amp
```

Startup prints:
- total discovered mp4 count
- train/val split sizes
- first 3 file paths

Split rules:
- deterministic shuffle using `--seed`
- `--max_videos N`: applied **after shuffle**, taking first `N`
- split then uses `--val_ratio`

## Overfit sanity mode

```bash
python train_video_ddpm.py \
  --data_root /path/to/data_root \
  --overfit_16 \
  --max_steps 200 \
  --batch_size 1
```

## Sample (DDIM + CFG)

Using dataset-derived condition frame:

```bash
python sample_video_ddpm.py \
  --ckpt ./outputs/train/last.pt \
  --data_root /path/to/data_root \
  --out_dir ./outputs/sample \
  --T 16 \
  --size 128 \
  --ddim_steps 50 \
  --ddim_eta 0 \
  --cfg_scale 2.0
```

Using custom condition image:

```bash
python sample_video_ddpm.py \
  --ckpt ./outputs/train/last.pt \
  --cond_image ./my_cond.png \
  --out_dir ./outputs/sample_custom \
  --T 16 --size 128 --ddim_steps 50
```

Outputs:
- `cond.png`: first-frame condition
- `sample.mp4`: generated grayscale clip
