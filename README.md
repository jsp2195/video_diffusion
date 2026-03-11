# Lightweight First-Frame-to-Future Grayscale Video Diffusion (PyTorch)

This repository is an MVP baseline for **future video continuation from a known first frame**:

\[
  p(\text{frames}_{1..T-1} \mid \text{frame}_0)
\]

## Active task (default everywhere)

- `frame_0` is conditioning-only input (`cond`, shape `[1,H,W]`)
- diffusion target is only future frames (`future`, shape `[1,T-1,H,W]`)
- the model does **not** denoise/regenerate frame 0
- sampling generates future frames only, then prepends the real condition frame

Final sampled clips are always:

`[frame_0, generated_frame_1, ..., generated_frame_{T-1}]`

## Baseline design choices

- Pixel-space grayscale diffusion (no latent stage)
- Lightweight conditional 3D U-Net (`models/video_unet3d.py`)
- Repeat+concat conditioning (`x_t` + tiled first frame)
- DDPM training with v-prediction
- DDIM sampling
- EMA for preview/inference
- AMP + grad clipping
- CFG conditioning dropout (default `0.08`)
- Fixed frame stride default `--frame_stride 1`

## MVP-oriented defaults

- `size=64`
- `T=8`
- `frame_stride=1`
- `base_channels=64`
- `channel_mults=1 2 4`
- `res_blocks=2`
- `noise_offset=0.0` (disabled by default)
- `dynamic_threshold=False` (disabled by default)

## Dataset expectations

The loader recursively discovers MP4s under `data_root`.

Returned sample contract:

- `cond`: `[1,H,W]` first frame
- `clip`: `[1,T,H,W]` full real clip (used to derive future target)

During training/validation, active target is always `clip[:, :, 1:]`.

## Commands

### 1) First overfit run

```bash
python train_video_ddpm.py \
  --data_root /path/to/data_root \
  --out_dir ./outputs/overfit_bair_mvp \
  --max_videos 64 \
  --size 64 \
  --T 8 \
  --frame_stride 1 \
  --batch_size 2 \
  --epochs 20 \
  --max_steps 500 \
  --cfg_drop_prob 0.08 \
  --vis_every 1 \
  --num_workers 2
```

### 2) Normal training run

```bash
python train_video_ddpm.py \
  --data_root /path/to/data_root \
  --out_dir ./outputs/train_bair_mvp \
  --size 64 \
  --T 8 \
  --frame_stride 1 \
  --batch_size 8 \
  --epochs 30 \
  --num_workers 4 \
  --lr 1e-4 \
  --cfg_drop_prob 0.08 \
  --vis_every 1 \
  --amp
```

### 3) Resume run

```bash
python train_video_ddpm.py \
  --data_root /path/to/data_root \
  --out_dir ./outputs/train_bair_mvp \
  --resume
```

### 4) Sampling run (EMA default)

```bash
python sample_video_ddpm.py \
  --ckpt ./outputs/train_bair_mvp/last.pt \
  --input_image ./cond.png \
  --out_dir ./outputs/sample_bair_mvp \
  --steps 40 \
  --eta 0.0 \
  --guidance_scale 1.8 \
  --device cuda
```

Sampling reconstructs architecture/task metadata from checkpoint config and generates **future-only** before prepending `frame_0`.

## Notes for BAIR-style MVP behavior

- This setup is tuned for simple structured motion and fast iteration.
- Keep horizons short (`T=8` or `T=12`) for first success.
- Proxy FVD remains a debug metric, not canonical FVD.

## Known limitations

- Compact model: limited long-range motion quality.
- Single-frame conditioning can drift on complex scenes or long horizons.
- Grayscale-only baseline prioritizes stability and low VRAM over visual richness.
