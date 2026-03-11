# Lightweight Endpoint-Conditioned Grayscale Video Diffusion (PyTorch)

This repository is an MVP baseline for **endpoint-conditioned middle-frame generation**:

\[
  p(\text{frames}_{1..T-2} \mid \text{frame}_0, \text{frame}_{T-1})
\]

## Active task (default everywhere)

- `frame_0` is a fixed conditioning start frame
- `frame_{T-1}` is a fixed conditioning end frame
- diffusion target is only middle frames (`frames_{1..T-2}`)
- the model does **not** regenerate either endpoint

Saved/generated clips are always assembled as:

`[exact_start, generated_middle..., exact_end]`

This endpoint-conditioned setup is intentionally easier than pure one-image-to-video and is aimed at reducing the "frame plus shimmer" failure mode on simple datasets (e.g. BAIR robot pushing).

## Model architecture (lightweight, stronger conditioning)

- Pixel-space grayscale diffusion (no latent stage)
- Compact 3D U-Net denoiser (`models/video_unet3d.py`)
- Shared lightweight multiscale conditioning encoder (`models/conditioning_encoder.py`) applied to both endpoints
- Endpoint feature fusion (`concat + 1x1 proj`) at each scale
- FiLM-style injection throughout the U-Net:
  - stem
  - each down block
  - bottleneck
  - each up block
- Temporal attention active at multiple levels by default (`temporal_attn_levels=1 2`)
- DDPM training with v-prediction + DDIM sampling
- EMA for preview/inference

## MVP defaults (single-GPU practical)

- `size=64`
- `T=8`
- `frame_stride=1`
- `base_channels=96`
- `channel_mults=1 2 4`
- `res_blocks=2`
- `temporal_attn_levels=1 2`
- `cfg_drop_prob=0.08`
- `temporal_loss_weight=0.0` (disabled)
- `noise_offset=0.0` (disabled)
- `dynamic_threshold=False` (disabled)

## Dataset expectations

The loader recursively discovers MP4s under `data_root` and returns grayscale clips:

- `cond`: first frame `[1,H,W]` (legacy field)
- `clip`: full clip `[1,T,H,W]`

Active training split is:

- `start = clip[:, :, 0]`
- `end = clip[:, :, -1]`
- `middle = clip[:, :, 1:-1]`

## Commands

### 1) First overfit run

```bash
python train_video_ddpm.py \
  --data_root /path/to/data_root \
  --out_dir ./outputs/overfit_endpoint_mvp \
  --max_videos 64 \
  --size 64 \
  --T 8 \
  --frame_stride 1 \
  --batch_size 2 \
  --epochs 20 \
  --max_steps 500 \
  --base_channels 96 \
  --channel_mults 1 2 4 \
  --temporal_attn_levels 1 2 \
  --cfg_drop_prob 0.08 \
  --temporal_loss_weight 0.0 \
  --vis_every 1 \
  --num_workers 2
```

### 2) Normal training run

```bash
python train_video_ddpm.py \
  --data_root /path/to/data_root \
  --out_dir ./outputs/train_endpoint_mvp \
  --size 64 \
  --T 8 \
  --frame_stride 1 \
  --batch_size 8 \
  --epochs 30 \
  --num_workers 4 \
  --lr 1e-4 \
  --base_channels 96 \
  --channel_mults 1 2 4 \
  --temporal_attn_levels 1 2 \
  --cfg_drop_prob 0.08 \
  --temporal_loss_weight 0.0 \
  --vis_every 1 \
  --amp
```

### 3) Resume run

```bash
python train_video_ddpm.py \
  --data_root /path/to/data_root \
  --out_dir ./outputs/train_endpoint_mvp \
  --resume
```

### 4) Sample generation run (EMA default)

```bash
python sample_video_ddpm.py \
  --ckpt ./outputs/train_endpoint_mvp/last.pt \
  --start_image ./start.png \
  --end_image ./end.png \
  --out_dir ./outputs/sample_endpoint_mvp \
  --steps 40 \
  --eta 0.0 \
  --guidance_scale 1.8 \
  --device cuda
```

(Alternative: omit `--start_image/--end_image` and provide `--data_root` to sample endpoint pairs from dataset clips.)

## Limitations

- This is still an MVP bridge/interpolation-style generator, not full unconstrained one-image-to-video.
- Conservative motion is still possible on difficult or noisy data.
- Long horizons and complex camera motion remain challenging.
