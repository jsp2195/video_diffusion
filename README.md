# Lightweight First-Frame Conditioned Grayscale Video Diffusion (PyTorch)

A practical baseline for generating short grayscale videos conditioned on a single first frame:

\[
  p(\text{video} \mid \text{first frame})
\]

This repo intentionally prioritizes **first success, speed, and debuggability** over SOTA quality.

## Active default pipeline

- Pixel-space grayscale diffusion (not latent diffusion)
- Compact conditional 3D U-Net (`models/video_unet3d.py`)
- First-frame conditioning by repeat + concat (`[B,2,T,H,W]` model input)
- DDPM training with **v-prediction**
- DDIM sampling
- EMA weights used for previews and inference by default
- AMP enabled by default (`--amp`, disable with `--no-amp`)
- Gradient clipping + classifier-free condition dropout
- Optional tiny temporal smoothness auxiliary loss

## Lightweight defaults

- `size=64`
- `T=8`
- `base_channels=64`
- `channel_mults=1 2 4`
- `res_blocks=2`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install decord imageio[ffmpeg] numpy pillow pyyaml matplotlib
```

## Dataset expectations

The loader recursively discovers MP4s under `data_root`.

Examples of valid layouts:

```text
/path/to/data_root/**/*.mp4
/path/to/data_root/videos_val/**/*.mp4
```

Returned dataset contract:

- `cond`: `[1,H,W]` first frame
- `clip`: `[1,T,H,W]` grayscale clip in `[-1,1]`

## First overfit run (recommended)

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

## Normal lightweight run

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

## Resume training

```bash
python train_video_ddpm.py \
  --data_root /path/to/data_root \
  --out_dir ./outputs/train_light \
  --resume
```

`--resume` loads `out_dir/last.pt` (model, EMA, optimizer, scheduler, scaler, step/epoch, and model config).

## Sampling from checkpoint (EMA default)

Using condition image:

```bash
python sample_video_ddpm.py \
  --ckpt ./outputs/train_light/last.pt \
  --input_image ./cond.png \
  --out_dir ./outputs/sample \
  --steps 40 \
  --eta 0.0 \
  --guidance_scale 1.8 \
  --device cuda
```

Using random dataset condition frame:

```bash
python sample_video_ddpm.py \
  --ckpt ./outputs/train_light/last.pt \
  --data_root /path/to/data_root \
  --out_dir ./outputs/sample \
  --steps 40 \
  --guidance_scale 1.8
```

Sampling **automatically reconstructs the trained architecture from checkpoint metadata** (`model_config`), so users do not need to re-enter architecture args.

Outputs:

- `cond.png`
- `sample.mp4`

## Metrics note (honesty)

`metrics/fvd.py` provides a lightweight **proxy** metric (`fvd_proxy`) using a small internal encoder.
It is useful for relative debugging trends only and is **not** canonical I3D-based FVD.

## Limitations

- Not SOTA quality; this is a compact baseline.
- Motion quality depends heavily on data diversity and clip count.
- Best workflow: overfit small subset first, then scale data and training length.
