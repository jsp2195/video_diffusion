import argparse
import os

import numpy as np
import torch
from PIL import Image

from data.kinetics_video_dataset import KineticsVideoDataset, discover_and_split_videos
from diffusion.schedule import DiffusionSchedule
from models.unet_video_cond import VideoUNetConditional
from utils.io import save_cond_png, save_mp4


def load_cond_from_image(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    gray = (gray / 127.5) - 1.0
    return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = VideoUNetConditional(in_channels=1, base_dim=64, cond_dim=256).to(device)
    state = ckpt["ema"] if ("ema" in ckpt and not args.use_raw_model) else ckpt["model"]
    model.load_state_dict(state)
    model.eval()

    if args.cond_image:
        cond = load_cond_from_image(args.cond_image, args.size)
    else:
        train_files, _ = discover_and_split_videos(args.data_root, args.val_ratio, args.seed, args.max_videos)
        ds = KineticsVideoDataset(train_files, num_frames=args.T, size=args.size)
        cond = ds[np.random.randint(0, len(ds))]["cond"].unsqueeze(0)

    cond = cond.to(device)
    diffusion = DiffusionSchedule(timesteps=args.timesteps, beta_schedule=args.beta_schedule).to(device)

    x = torch.randn(1, 1, args.T, args.size, args.size, device=device)

    ddim_times = torch.linspace(args.timesteps - 1, 0, args.ddim_steps, device=device).long()
    for i in range(len(ddim_times)):
        t = ddim_times[i].repeat(1)
        t_prev = ddim_times[i + 1].repeat(1) if i < len(ddim_times) - 1 else torch.tensor([-1], device=device)

        zeros_cond = torch.zeros_like(cond)
        eps_u = model(x, t, zeros_cond)
        eps_c = model(x, t, cond)
        eps = eps_u + args.cfg_scale * (eps_c - eps_u)

        x = diffusion.ddim_step(x, eps, t, t_prev, eta=args.ddim_eta)

    os.makedirs(args.out_dir, exist_ok=True)
    save_cond_png(cond.unsqueeze(2), os.path.join(args.out_dir, "cond.png"))
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
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="cosine", choices=["cosine", "linear"])
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--ddim_eta", type=float, default=0.0)
    p.add_argument("--cfg_scale", type=float, default=2.0)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--use_raw_model", action="store_true")
    args = p.parse_args()

    if args.cond_image is None and args.data_root is None:
        raise ValueError("Provide --cond_image or --data_root")
    return args


if __name__ == "__main__":
    sample(parse_args())
