# Grayscale Latent Image-to-Video Diffusion (Default Path)

This repository now defaults to a **Stable-Video-Diffusion-inspired latent pipeline** for **native grayscale first-frame-to-video generation**.

## Default architecture

- **Stage A:** Temporal grayscale `AutoencoderKL` (`f=8`, `z_channels=4`)
- **Stage B:** Latent grayscale video diffusion with `VideoUNet`
- **Conditioning stack:**
  - latent concat from encoded first frame
  - learned grayscale image token embedder
  - scalar embeddings (`fps_id`, `motion_bucket_id`, `cond_aug`)
- **Diffusion math:** sigma-based denoiser/sampler with latent **v-prediction**
- **Sampler:** Euler

The old pixel-space DDPM scripts remain in repo as a legacy baseline (`train_video_ddpm.py`, `sample_video_ddpm.py`), but they are no longer the documented default path.

## Data layout (unchanged)

Dataset discovery is still based on:

```text
data_root/
  videos_val/
    .../*.mp4
```

No format conversion is required.

## Stage A: train grayscale temporal autoencoder

1) Edit `configs/training/ae_stage.yaml` and set `data.root`.

2) Run:

```bash
python train_ae.py --config configs/training/ae_stage.yaml
```

Checkpoint: `outputs/ae/ae_last.pt`

## Stage B: train latent video diffusion

1) Edit `configs/training/ldm_stage.yaml` and set:
- `data.root`
- `training.ae_ckpt`

2) Run:

```bash
python train_latent_diffusion.py --config configs/training/ldm_stage.yaml
```

Checkpoint: `outputs/ldm/ldm_last.pt`

## Sampling (first-frame image-to-video)

```bash
python sample_i2v.py \
  --config configs/sampling/euler_gray_i2v.yaml \
  --cond_image /path/to/cond_frame_gray.png \
  --ae_ckpt outputs/ae/ae_last.pt \
  --ldm_ckpt outputs/ldm/ldm_last.pt \
  --out outputs/sample_i2v.mp4
```

## Notes

- Pipeline is grayscale-native end-to-end (model inputs/outputs remain 1-channel).
- No pretrained components are used.
- The new default path does **not** hard-overwrite frame 0 during denoising.
- Optional final first-frame replacement is done only once at output time in `sample_i2v.py`.
