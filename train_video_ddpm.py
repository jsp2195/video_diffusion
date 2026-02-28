import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _denorm_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    return x.numpy()


def _frame_indices(t_len: int):
    return [0, t_len // 4, t_len // 2, (3 * t_len) // 4, t_len - 1]


@torch.no_grad()
def _ddim_sample_with_cfg(model: torch.nn.Module, diffusion: DiffusionSchedule, cond: torch.Tensor, t_len: int, size: int, steps: int, guidance_scale: float, eta: float = 0.0):
    device = cond.device
    x = torch.randn(1, 1, t_len, size, size, device=device)
    ddim_times = torch.linspace(diffusion.timesteps - 1, 0, steps, device=device).long()

    for i in range(len(ddim_times)):
        t = ddim_times[i].view(1)
        t_prev = ddim_times[i + 1].view(1) if i < len(ddim_times) - 1 else torch.tensor([-1], device=device)

        zeros_cond = torch.zeros_like(cond)
        eps_u = model(x, t, zeros_cond)
        eps_c = model(x, t, cond)
        eps = eps_u + guidance_scale * (eps_c - eps_u)
        x = diffusion.ddim_step(x, eps, t, t_prev, eta=eta)
    return x


@torch.no_grad()
def run_epoch_visualization(
    args,
    epoch: int,
    ema_model: torch.nn.Module,
    diffusion: DiffusionSchedule,
    fixed_val_ds: KineticsVideoDataset,
    fixed_idx: int,
    out_dir: str,
    device: torch.device,
):
    sample = fixed_val_ds[fixed_idx]
    cond = sample["cond"].unsqueeze(0).to(device)  # [1,1,H,W]
    gt_clip = sample["clip"].unsqueeze(0).to(device)  # [1,1,T,H,W]

    gen_clip = _ddim_sample_with_cfg(
        model=ema_model,
        diffusion=diffusion,
        cond=cond,
        t_len=args.T,
        size=args.size,
        steps=args.vis_steps,
        guidance_scale=args.vis_guidance_scale,
        eta=0.0,
    )

    frame_ids = _frame_indices(args.T)
    gt_np = _denorm_to_uint8(gt_clip)[0, 0]  # [T,H,W]
    gen_np = _denorm_to_uint8(gen_clip)[0, 0]  # [T,H,W]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for col, fid in enumerate(frame_ids):
        axes[0, col].imshow(gt_np[fid], cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_axis_off()
        axes[0, col].set_title(f"GT t={fid}")

        axes[1, col].imshow(gen_np[fid], cmap="gray", vmin=0, vmax=255)
        axes[1, col].set_axis_off()
        axes[1, col].set_title(f"GEN t={fid}")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"epoch_{epoch:04d}_comparison.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    mp4_path = os.path.join(out_dir, f"epoch_{epoch:04d}_sample.mp4")
    save_mp4(gen_clip, mp4_path, fps=8)

    print(f"visualization saved: {fig_path}")
    print(f"sample video saved: {mp4_path}")


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

    # deterministic fixed validation sample for epoch-wise visualization
    fixed_rng = random.Random(args.seed)
    fixed_idx = fixed_rng.randrange(len(val_ds))
    fixed_path = val_ds.files[fixed_idx]
    print(f"Fixed visualization validation sample idx: {fixed_idx}")
    print(f"Fixed visualization validation sample path: {fixed_path}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = VideoUNetConditional(in_channels=1, base_dim=64, cond_dim=256).to(device)
    ema = EMA(model, decay=args.ema_decay)

    diffusion = DiffusionSchedule(timesteps=args.timesteps, beta_schedule=args.beta_schedule).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(train_loader) * args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    os.makedirs(args.out_dir, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            cond = batch["cond"].to(device)
            clip = batch["clip"].to(device)

            if args.cfg_drop_prob > 0:
                drop = (torch.rand(cond.shape[0], device=device) < args.cfg_drop_prob).float().view(-1, 1, 1, 1)
                cond = cond * (1.0 - drop)

            t = diffusion.sample_timesteps(cond.shape[0], device)
            noisy, noise = diffusion.forward_noise(clip, t)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(noisy, t, cond)
                loss = F.mse_loss(pred, noise)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            global_step += 1

            if args.shape_check and global_step == 1:
                print(f"shape_check cond={tuple(cond.shape)} clip={tuple(clip.shape)} noisy={tuple(noisy.shape)} pred={tuple(pred.shape)}")

            if global_step % args.log_every == 0:
                print(f"epoch={epoch} step={global_step} loss={loss.item():.6f}")

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

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                cond = batch["cond"].to(device)
                clip = batch["clip"].to(device)
                t = diffusion.sample_timesteps(cond.shape[0], device)
                noisy, noise = diffusion.forward_noise(clip, t)
                pred = model(noisy, t, cond)
                val_losses.append(F.mse_loss(pred, noise).item())
        print(f"val_loss={np.mean(val_losses):.6f}")

        if (epoch + 1) % args.vis_every == 0:
            try:
                run_epoch_visualization(
                    args=args,
                    epoch=epoch + 1,
                    ema_model=ema.ema_model,
                    diffusion=diffusion,
                    fixed_val_ds=val_ds,
                    fixed_idx=fixed_idx,
                    out_dir=args.out_dir,
                    device=device,
                )
            except Exception as e:
                print(f"[warn] visualization failed at epoch {epoch + 1}: {e}")

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
    p.add_argument("--ema_decay", type=float, default=0.999)
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
