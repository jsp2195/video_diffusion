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


MODEL_TYPE = "video_unet3d_cond"
PREDICTION_TARGET = "v"
TASK_MODE = "future_only_from_first_frame"


def load_cond_from_image(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    gray = (gray / 127.5) - 1.0
    return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)


def infer_model_config_from_checkpoint(ckpt: dict) -> dict:
    cfg = ckpt.get("model_config")
    if cfg:
        return cfg

    train_cfg = ckpt.get("config", {})
    return {
        "model_type": MODEL_TYPE,
        "in_channels": 1,
        "cond_channels": 1,
        "base_channels": int(train_cfg.get("base_channels", 96)),
        "channel_mult": list(train_cfg.get("channel_mults", [1, 2, 4])),
        "num_res_blocks": int(train_cfg.get("res_blocks", 2)),
        "temporal_attn_levels": list(train_cfg.get("temporal_attn_levels", [1, 2])),
        "cond_encoder_type": str(train_cfg.get("cond_encoder_type", "multiscale_pyramid")),
        "cond_injection_mode": str(train_cfg.get("cond_injection_mode", "add")),
        "prediction_target": str(train_cfg.get("prediction_target", PREDICTION_TARGET)),
        "task_mode": str(train_cfg.get("task_mode", TASK_MODE)),
    }


def build_model_from_cfg(model_cfg: dict) -> VideoUNet3DConditional:
    if model_cfg.get("model_type") != MODEL_TYPE:
        raise ValueError(f"Unsupported model type: {model_cfg.get('model_type')}")
    if model_cfg.get("prediction_target", PREDICTION_TARGET) != PREDICTION_TARGET:
        raise ValueError("This sampler expects a v-prediction checkpoint.")
    if model_cfg.get("task_mode", TASK_MODE) != TASK_MODE:
        raise ValueError(f"Unsupported checkpoint task_mode={model_cfg.get('task_mode')}. Expected {TASK_MODE}.")
    if model_cfg.get("cond_encoder_type", "multiscale_pyramid") != "multiscale_pyramid":
        raise ValueError("Unsupported checkpoint cond_encoder_type. Expected multiscale_pyramid.")
    if model_cfg.get("cond_injection_mode", "add") != "add":
        raise ValueError("Unsupported checkpoint cond_injection_mode. Expected add.")

    return VideoUNet3DConditional(
        in_channels=int(model_cfg.get("in_channels", 1)),
        cond_channels=int(model_cfg.get("cond_channels", 1)),
        base_channels=int(model_cfg["base_channels"]),
        channel_mult=tuple(model_cfg["channel_mult"]),
        num_res_blocks=int(model_cfg["num_res_blocks"]),
        temporal_attn_levels=tuple(model_cfg.get("temporal_attn_levels", [1, 2])),
        cond_injection_mode=str(model_cfg.get("cond_injection_mode", "add")),
    )


@torch.no_grad()
def sample(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device)
    model_cfg = infer_model_config_from_checkpoint(ckpt)
    model = build_model_from_cfg(model_cfg).to(device)

    if args.use_raw_model:
        model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model")))
    else:
        if "ema_state_dict" in ckpt:
            model.load_state_dict(ckpt["ema_state_dict"]["model"])
        elif "ema" in ckpt and isinstance(ckpt["ema"], dict) and "model" in ckpt["ema"]:
            model.load_state_dict(ckpt["ema"]["model"])
        else:
            raise ValueError("Checkpoint has no EMA state. Use --use_raw_model to force raw weights.")
    model.eval()

    if args.T is None:
        args.T = int(model_cfg.get("T", ckpt.get("config", {}).get("T", 8)))
    if args.T < 2:
        raise ValueError("Resolved T must be >= 2 for future-only sampling.")
    if args.frame_stride is None:
        args.frame_stride = int(model_cfg.get("frame_stride", ckpt.get("config", {}).get("frame_stride", 1)))

    if args.input_image:
        cond = load_cond_from_image(args.input_image, args.size)
    else:
        train_files, _, _, _ = discover_and_split_videos(args.data_root, args.val_ratio, args.seed, args.max_videos)
        ds = KineticsVideoDataset(train_files, num_frames=args.T, frame_stride=args.frame_stride, size=args.size)
        cond = ds[np.random.randint(0, len(ds))]["cond"].unsqueeze(0)

    cond = cond.to(device)
    diffusion = DiffusionSchedule(args.timesteps, schedule=args.beta_schedule).to(device)

    future_len = args.T - 1
    x = torch.randn(1, 1, future_len, args.size, args.size, device=device)
    ddim_times = torch.linspace(args.timesteps - 1, 0, args.steps, device=device).long()

    for i in range(len(ddim_times)):
        t = ddim_times[i].view(1)
        t_prev = ddim_times[i + 1].view(1) if i < len(ddim_times) - 1 else torch.tensor([-1], device=device)

        zeros_cond = torch.zeros_like(cond)
        v_uncond = model(x, t, zeros_cond)
        v_cond = model(x, t, cond)
        v = v_uncond + args.guidance_scale * (v_cond - v_uncond)

        x = diffusion.ddim_step_from_v(x, v, t, t_prev, eta=args.eta, dynamic_threshold=args.dynamic_threshold)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.out_path:
        out_mp4 = args.out_path
        os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
        cond_path = os.path.join(os.path.dirname(out_mp4) or ".", "cond.png")
    else:
        out_mp4 = os.path.join(args.out_dir, "sample.mp4")
        cond_path = os.path.join(args.out_dir, "cond.png")

    full_clip = torch.cat([cond.unsqueeze(2), x], dim=2)
    save_cond_png(cond, cond_path)
    save_mp4(full_clip, out_mp4, fps=args.fps)
    print(f"saved: {cond_path}")
    print(f"saved: {out_mp4}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--val_ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_videos", type=int, default=None)
    p.add_argument("--input_image", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/sample")
    p.add_argument("--out_path", type=str, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--frame_stride", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="cosine", choices=["cosine", "linear"])
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--guidance_scale", type=float, default=1.8)
    p.add_argument("--dynamic_threshold", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_raw_model", action="store_true")

    # Backward-compatible aliases.
    p.add_argument("--cond_image", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--ddim_steps", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--ddim_eta", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--cfg_scale", type=float, default=None, help=argparse.SUPPRESS)

    args = p.parse_args()

    if args.input_image is None and args.cond_image is not None:
        args.input_image = args.cond_image
    if args.ddim_steps is not None:
        args.steps = args.ddim_steps
    if args.ddim_eta is not None:
        args.eta = args.ddim_eta
    if args.cfg_scale is not None:
        args.guidance_scale = args.cfg_scale

    if args.input_image is None and args.data_root is None:
        raise ValueError("Provide --input_image or --data_root")
    return args


if __name__ == "__main__":
    sample(parse_args())
