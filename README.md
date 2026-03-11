# Lightweight Multi-Endpoint-Context RGB Video Diffusion (PyTorch)

This repository is an MVP baseline for **motion-conditioned middle-frame generation**:

\[
  p(\text{frames}_{K..T-K-1} \mid \text{first }K\text{ frames},\ \text{last }K\text{ frames})
\]

Default: `K = 2` (`--endpoint_context 2`).

For `T=8`, the model observes frames `[0,1]` and `[6,7]`, and generates frames `[2,3,4,5]`.

## Active task (default everywhere)

- First `K` frames are fixed observed start context
- Last `K` frames are fixed observed end context
- Diffusion target is only the center span
- Observed context frames are never denoised/regenerated

Saved/generated clips are assembled exactly as:

`[exact_start_context, generated_middle..., exact_end_context]`

This setup is intentionally more constrained than one-image-to-video to reduce shimmer and endpoint morphing, and encourage visible short-horizon motion.

Grayscale remains available as an optional mode (`--color_mode gray`), but RGB is the active MVP default for stronger object/background and identity cues.

## Model architecture (lightweight, stronger conditioning)

- Pixel-space RGB diffusion by default (no latent stage)
- Compact 3D U-Net denoiser (`models/video_unet3d.py`)
- Shared multiscale conditioning encoder (`models/conditioning_encoder.py`) applied to both start/end context clips
- Endpoint feature fusion (`concat + 1x1 projection`) per scale
- FiLM-style modulation at:
  - stem
  - each down block
  - bottleneck
  - each up block
- Temporal attention active at multiple levels by default (`temporal_attn_levels=1 2`)
- DDPM training + DDIM sampling + EMA

## MVP defaults (single-GPU practical, RGB default)

- `size=64`
- `T=8`
- `endpoint_context=2`
- `frame_stride=1`
- `base_channels=96`
- `channel_mults=1 2 4`
- `res_blocks=2`
- `temporal_attn_levels=1 2`
- `cfg_drop_prob=0.08`
- `temporal_loss_weight=0.05` (enabled)
- `noise_offset=0.0` (disabled)
- `dynamic_threshold=False` (disabled)
- `color_mode=rgb` (default, recommended)

## Dataset expectations

The loader recursively discovers MP4s under `data_root` and returns **RGB clips by default**:

- `clip`: full clip `[3,T,H,W]` for default RGB (`[1,T,H,W]` only if `--color_mode gray`)

Active split used in train/val/sample:

- `start_context = clip[:, :, :K]`
- `middle_target = clip[:, :, K:-K]`
- `end_context = clip[:, :, -K:]`

`T` must satisfy `T > 2*K`.

## Commands

### 1) First overfit run

```bash
python train_video_ddpm.py \
  --data_root /path/to/data_root \
  --out_dir ./outputs/overfit_motion_ctx2 \
  --max_videos 64 \
  --size 64 \
  --T 8 \
  --endpoint_context 2 \
  --frame_stride 1 \
  --color_mode rgb \
  --batch_size 2 \
  --epochs 20 \
  --max_steps 500 \
  --base_channels 96 \
  --channel_mults 1 2 4 \
  --temporal_attn_levels 1 2 \
  --cfg_drop_prob 0.08 \
  --temporal_loss_weight 0.05 \
  --vis_every 1 \
  --num_workers 2
```

### 2) Normal training run

```bash
python train_video_ddpm.py \
  --data_root /path/to/data_root \
  --out_dir ./outputs/train_motion_ctx2 \
  --size 64 \
  --T 8 \
  --endpoint_context 2 \
  --frame_stride 1 \
  --color_mode rgb \
  --batch_size 8 \
  --epochs 30 \
  --num_workers 4 \
  --lr 1e-4 \
  --base_channels 96 \
  --channel_mults 1 2 4 \
  --temporal_attn_levels 1 2 \
  --cfg_drop_prob 0.08 \
  --temporal_loss_weight 0.05 \
  --vis_every 1 \
  --amp
```

### 3) Resume run

```bash
python train_video_ddpm.py \
  --data_root /path/to/data_root \
  --out_dir ./outputs/train_motion_ctx2 \
  --resume
```

### 4) Sample generation run (EMA default)

```bash
python sample_video_ddpm.py \
  --ckpt ./outputs/train_motion_ctx2/last.pt \
  --start_images ./start_0.png ./start_1.png \
  --end_images ./end_0.png ./end_1.png \
  --endpoint_context 2 \
  --color_mode rgb \
  --out_dir ./outputs/sample_motion_ctx2 \
  --steps 40 \
  --eta 0.0 \
  --guidance_scale 1.8 \
  --device cuda
```

(Alternative: omit `--start_images/--end_images` and pass `--data_root` to draw context clips from dataset videos.)

## Limitations

- This is still an MVP bridge generator, not unconstrained one-image-to-video.
- Conservative or partial motion can still appear on difficult scenes.
- Long horizons and complex camera movement remain challenging.
