import argparse
import os
import random

import numpy as np
import torch
from PIL import Image

from data.kinetics_video_dataset import KineticsVideoDataset, discover_and_split_videos
from diffusion.schedule import DiffusionSchedule
from models.video_unet3d import VideoUNet3DConditional
from utils.io import save_cond_png, save_mp4


def load_cond_from_image(path: str, size: int, channels: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    if channels == 1:
        arr = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2])[..., None]
    arr = (arr / 127.5) - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = VideoUNet3DConditional(
        in_channels=args.channels,
        base_channels=128,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_heads=8,
        max_temporal_length=max(32, args.T),
    ).to(device)

    if not args.use_raw_model:
        if "ema_model" in ckpt:
            model.load_state_dict(ckpt["ema_model"])
        elif "ema_state_dict" in ckpt:
            model.load_state_dict(ckpt["ema_state_dict"]["model"])
        elif "ema" in ckpt and isinstance(ckpt["ema"], dict) and "model" in ckpt["ema"]:
            model.load_state_dict(ckpt["ema"]["model"])
        else:
            model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model")))
    else:
        model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model")))
    model.eval()

    if args.cond_image:
        cond = load_cond_from_image(args.cond_image, args.size, args.channels)
    else:
        train_files, _, _, _ = discover_and_split_videos(args.data_root, args.val_ratio, args.seed, args.max_videos)
        ds = KineticsVideoDataset(train_files, num_frames=args.T, size=args.size, channels=args.channels)
        cond = ds[np.random.randint(0, len(ds))]["cond"].unsqueeze(0)

    cond = cond.to(device)
    diffusion = DiffusionSchedule(args.timesteps, schedule=args.beta_schedule).to(device)

    x = torch.randn(1, args.channels, args.T, args.size, args.size, device=device)
    ddim_times = torch.linspace(args.timesteps - 1, 0, args.ddim_steps, device=device).long()

    for i in range(len(ddim_times)):
        t = ddim_times[i].view(1)
        t_prev = ddim_times[i + 1].view(1) if i < len(ddim_times) - 1 else torch.tensor([-1], device=device)

        zeros_cond = torch.zeros_like(cond)
        v_uncond = model(x, t, zeros_cond)
        v_cond = model(x, t, cond)
        v = v_uncond + args.guidance_scale * (v_cond - v_uncond)

        x = diffusion.ddim_step_from_v(x, v, t, t_prev, eta=args.ddim_eta)
        x[:, :, 0] = cond

    os.makedirs(args.out_dir, exist_ok=True)
    save_cond_png(cond, os.path.join(args.out_dir, "cond.png"))
    save_mp4(x, os.path.join(args.out_dir, "sample.mp4"), fps=args.fps)
    print(f"saved: {os.path.join(args.out_dir, 'cond.png')}")
    print(f"saved: {os.path.join(args.out_dir, 'sample.mp4')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--val_ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_videos", type=int, default=None)
    p.add_argument("--cond_image", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/sample")
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="cosine", choices=["cosine", "linear"])
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--ddim_eta", type=float, default=0.0)
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--cfg_scale", type=float, default=None)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--use_raw_model", action="store_true")
    args = p.parse_args()

    if args.cfg_scale is not None:
        args.guidance_scale = args.cfg_scale
    if args.cond_image is None and args.data_root is None:
        raise ValueError("Provide --cond_image or --data_root")
    return args


if __name__ == "__main__":
    sample(parse_args())
