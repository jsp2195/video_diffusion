import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.kinetics_video_dataset import KineticsVideoDataset, discover_and_split_videos
from diffusion.schedule import DiffusionSchedule
from models.unet_video_cond import VideoUNetConditional
from utils.ema import EMA
from utils.io import save_mp4


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def sample_preview(ema_model, diffusion, cond, steps: int, guidance_scale: float, out_path: str):
    device = cond.device
    t_len = cond.shape[-2] if cond.ndim == 5 else None
    # cond expected [1,1,H,W]
    h, w = cond.shape[-2:]
    t_len = 16 if t_len is None else t_len
    x = torch.randn(1, 1, t_len, h, w, device=device)
    ddim_times = torch.linspace(diffusion.timesteps - 1, 0, steps, device=device).long()

    for i in range(len(ddim_times)):
        t = ddim_times[i].view(1)
        t_prev = ddim_times[i + 1].view(1) if i < len(ddim_times) - 1 else torch.tensor([-1], device=device)
        zeros_cond = torch.zeros_like(cond)
        v_u = ema_model(x, t, zeros_cond)
        v_c = ema_model(x, t, cond)
        v = v_u + guidance_scale * (v_c - v_u)
        x = diffusion.ddim_step_from_v(x, v, t, t_prev, eta=0.0)

    save_mp4(x, out_path, fps=8)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files, val_files, _, _ = discover_and_split_videos(
        data_root=args.data_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_videos=args.max_videos,
    )
    if args.overfit_16:
        train_files = train_files[:16]

    train_ds = KineticsVideoDataset(train_files, num_frames=args.T, size=args.size)
    val_ds = KineticsVideoDataset(val_files, num_frames=args.T, size=args.size)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = VideoUNetConditional(in_channels=1, base_channels=64).to(device)
    ema = EMA(model, decay=args.ema_decay, update_after_step=1000)
    diffusion = DiffusionSchedule(timesteps=args.timesteps).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(train_loader) * args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    os.makedirs(args.out_dir, exist_ok=True)
    global_step = 0

    fixed_rng = random.Random(args.seed)
    fixed_idx = fixed_rng.randrange(len(val_ds))

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            cond = batch["cond"].to(device)  # [B,1,H,W]
            clip = batch["clip"].to(device)  # [B,1,T,H,W]

            if args.cfg_drop_prob > 0:
                mask = (torch.rand(cond.shape[0], device=device) < args.cfg_drop_prob).float().view(-1, 1, 1, 1)
                cond = cond * (1.0 - mask)

            t = diffusion.sample_timesteps(cond.shape[0], device)
            x_t, noise = diffusion.forward_noise(clip, t)
            v_target = diffusion.velocity_target(clip, noise, t)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred_v = model(x_t, t, cond)
                loss = F.mse_loss(pred_v, v_target)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            global_step += 1

            if args.shape_check and global_step == 1:
                print(f"shape_check cond={tuple(cond.shape)} clip={tuple(clip.shape)} x_t={tuple(x_t.shape)} pred_v={tuple(pred_v.shape)}")

            if global_step % args.log_every == 0:
                print(f"epoch={epoch+1} step={global_step} loss={loss.item():.6f}")

            if global_step % args.save_every == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "step": global_step,
                    "args": vars(args),
                }
                torch.save(ckpt, os.path.join(args.out_dir, f"ckpt_{global_step}.pt"))

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch in val_loader:
                cond = batch["cond"].to(device)
                clip = batch["clip"].to(device)
                t = diffusion.sample_timesteps(cond.shape[0], device)
                x_t, noise = diffusion.forward_noise(clip, t)
                v_target = diffusion.velocity_target(clip, noise, t)
                pred_v = model(x_t, t, cond)
                val_losses.append(F.mse_loss(pred_v, v_target).item())
            print(f"val_loss={np.mean(val_losses):.6f}")

            if (epoch + 1) % args.vis_every == 0:
                preview_sample = val_ds[fixed_idx]
                preview_cond = preview_sample["cond"].unsqueeze(0).to(device)
                preview_path = os.path.join(args.out_dir, f"epoch_{epoch+1:04d}_sample.mp4")
                sample_preview(ema.ema_model, diffusion, preview_cond, steps=args.vis_steps, guidance_scale=args.vis_guidance_scale, out_path=preview_path)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    final_ckpt = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "step": global_step,
        "args": vars(args),
    }
    torch.save(final_ckpt, os.path.join(args.out_dir, "last.pt"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--val_ratio", type=float, default=0.01)
    p.add_argument("--out_dir", type=str, default="outputs/train")
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="cosine", choices=["cosine", "linear"])
    p.add_argument("--cfg_drop_prob", type=float, default=0.1)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_videos", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--shape_check", action="store_true")
    p.add_argument("--overfit_16", action="store_true")
    p.add_argument("--vis_every", type=int, default=1)
    p.add_argument("--vis_guidance_scale", type=float, default=1.5)
    p.add_argument("--vis_steps", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
