import argparse
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.kinetics_video_dataset import KineticsVideoDataset, discover_and_split_videos
from diffusion.schedule import DiffusionSchedule
from metrics.fvd import compute_fvd
from models.unet_video_cond import VideoUNetConditional
from utils.config import parse_with_config
from utils.diagnostics import save_training_curves
from utils.distributed import cleanup_distributed, get_rank, get_world_size, init_distributed, is_main_process
from utils.ema import EMA
from utils.io import save_mp4
from utils.logger import TrainLogger


def set_seed(seed: int, rank: int = 0):
    full_seed = seed + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    torch.cuda.manual_seed_all(full_seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


@torch.no_grad()
def sample_video(model, diffusion, cond, t_len: int, size: int, steps: int, guidance_scale: float, eta: float = 0.0):
    device = cond.device
    x = torch.randn(1, 1, t_len, size, size, device=device)
    ddim_times = torch.linspace(diffusion.timesteps - 1, 0, steps, device=device).long()
    for i in range(len(ddim_times)):
        t = ddim_times[i].view(1)
        t_prev = ddim_times[i + 1].view(1) if i < len(ddim_times) - 1 else torch.tensor([-1], device=device)
        zeros_cond = torch.zeros_like(cond)
        v_u = model(x, t, zeros_cond)
        v_c = model(x, t, cond)
        v = v_u + guidance_scale * (v_c - v_u)
        x = diffusion.ddim_step_from_v(x, v, t, t_prev, eta=eta)
    return x


@torch.no_grad()
def evaluate_fvd(ema_model, diffusion, val_ds, args, device):
    num_eval = min(args.num_eval_videos, len(val_ds))
    real_videos = []
    gen_videos = []
    rng = random.Random(args.seed)
    indices = [rng.randrange(len(val_ds)) for _ in range(num_eval)]

    for idx in indices:
        sample = val_ds[idx]
        cond = sample["cond"].unsqueeze(0).to(device)
        real = sample["clip"].unsqueeze(0).to(device)
        gen = sample_video(
            ema_model,
            diffusion,
            cond,
            t_len=args.T,
            size=args.size,
            steps=args.vis_steps,
            guidance_scale=args.vis_guidance_scale,
            eta=0.0,
        )
        real_videos.append(real)
        gen_videos.append(gen)

    real_batch = torch.cat(real_videos, dim=0)
    gen_batch = torch.cat(gen_videos, dim=0)
    return compute_fvd(real_batch, gen_batch, device)


def train(args):
    distributed, rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed, rank)

    train_files, val_files, _, _ = discover_and_split_videos(
        data_root=args.data_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_videos=args.max_videos,
    )
    if args.overfit_16:
        train_files = train_files[:16]

    train_ds = KineticsVideoDataset(
        train_files,
        num_frames=args.T,
        size=args.size,
        cache_videos=args.cache_videos,
    )
    val_ds = KineticsVideoDataset(
        val_files,
        num_frames=args.T,
        size=args.size,
        cache_videos=args.cache_videos,
    )

    g = torch.Generator()
    g.manual_seed(args.seed + rank)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )

    model = VideoUNetConditional(in_channels=1, base_channels=32).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, output_device=local_rank if torch.cuda.is_available() else None, find_unused_parameters=False)

    ema = EMA(unwrap_model(model), decay=args.ema_decay, update_after_step=1000)
    diffusion = DiffusionSchedule(timesteps=args.timesteps).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(train_loader) * args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    logger = TrainLogger(enabled=args.tensorboard and is_main_process(), log_dir=args.log_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    global_step = 0
    fixed_rng = random.Random(args.seed)
    fixed_idx = fixed_rng.randrange(len(val_ds)) if len(val_ds) > 0 else None

    history = {"train_loss": [], "val_loss": [], "fvd": [], "fvd_epochs": []}
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_losses = []

        for batch in train_loader:
            cond = batch["cond"].to(device, non_blocking=True)
            clip = batch["clip"].to(device, non_blocking=True)

            if args.cfg_drop_prob > 0:
                mask = (torch.rand(cond.shape[0], device=device) < args.cfg_drop_prob).float().view(-1, 1, 1, 1)
                cond = cond * (1.0 - mask)

            t = diffusion.sample_timesteps(cond.shape[0], device)
            x_t, noise = diffusion.forward_noise(clip, t)
            v_target = diffusion.velocity_target(clip, noise, t)

            amp_ctx = torch.cuda.amp.autocast(enabled=args.amp) if torch.cuda.is_available() else nullcontext()
            with amp_ctx:
                pred_v = model(x_t, t, cond)
                loss = F.mse_loss(pred_v, v_target)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(unwrap_model(model))
            global_step += 1
            epoch_losses.append(loss.item())

            if args.shape_check and global_step == 1 and is_main_process():
                print(f"shape_check cond={tuple(cond.shape)} clip={tuple(clip.shape)} x_t={tuple(x_t.shape)} pred_v={tuple(pred_v.shape)}")

            if global_step % args.log_every == 0 and is_main_process():
                print(f"epoch={epoch+1} step={global_step} loss={loss.item():.6f}")
                logger.add_scalar("train/loss", float(loss.item()), global_step)
                logger.add_scalar("learning_rate", float(opt.param_groups[0]["lr"]), global_step)

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        train_loss_epoch = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history["train_loss"].append(train_loss_epoch)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                cond = batch["cond"].to(device, non_blocking=True)
                clip = batch["clip"].to(device, non_blocking=True)
                t = diffusion.sample_timesteps(cond.shape[0], device)
                x_t, noise = diffusion.forward_noise(clip, t)
                v_target = diffusion.velocity_target(clip, noise, t)
                pred_v = model(x_t, t, cond)
                val_losses.append(F.mse_loss(pred_v, v_target).item())

        val_loss_epoch = float(np.mean(val_losses)) if val_losses else 0.0

        if distributed:
            t_train = torch.tensor([train_loss_epoch], device=device)
            t_val = torch.tensor([val_loss_epoch], device=device)
            dist.all_reduce(t_train, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_val, op=dist.ReduceOp.SUM)
            train_loss_epoch = (t_train / world_size).item()
            val_loss_epoch = (t_val / world_size).item()

        history["val_loss"].append(val_loss_epoch)

        if is_main_process():
            print(f"epoch={epoch+1} train_loss={train_loss_epoch:.6f} val_loss={val_loss_epoch:.6f}")
            model_unwrapped = unwrap_model(model)
            ckpt = {
                "model_state_dict": model_unwrapped.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "step": global_step,
                "epoch": epoch + 1,
                "config": vars(args),
            }

            # save most recent checkpoint
            torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))

            # save best checkpoint
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            
            logger.add_scalar("val/loss", val_loss_epoch, epoch + 1)

            if (epoch + 1) % args.vis_every == 0 and fixed_idx is not None:
                preview_sample = val_ds[fixed_idx]
                preview_cond = preview_sample["cond"].unsqueeze(0).to(device)
                preview_video = sample_video(
                    ema.ema_model if hasattr(ema, "ema_model") else ema,
                    diffusion,
                    preview_cond,
                    t_len=args.T,
                    size=args.size,
                    steps=args.vis_steps,
                    guidance_scale=args.vis_guidance_scale,
                    eta=0.0,
                )
                preview_path = os.path.join(args.out_dir, f"epoch_{epoch+1:04d}_sample.mp4")
                save_mp4(preview_video[0], preview_path, fps=8)
                logger.add_video("preview/video", ((preview_video.clamp(-1, 1) + 1.0) / 2.0).permute(0, 2, 1, 3, 4), epoch + 1, fps=8)

            if args.eval_fvd_every > 0 and (epoch + 1) % args.eval_fvd_every == 0:
                ema_model = ema.ema_model if hasattr(ema, "ema_model") else ema
                fvd_score = evaluate_fvd(ema_model, diffusion, val_ds, args, device)
                history["fvd"].append(fvd_score)
                history["fvd_epochs"].append(epoch + 1)
                print(f"epoch={epoch+1}, fvd_score={fvd_score:.4f}")
                logger.add_scalar("metrics/fvd", fvd_score, epoch + 1)

            save_training_curves(history, args.out_dir)

        if distributed:
            dist.barrier()

        if args.max_steps > 0 and global_step >= args.max_steps:
            break


    logger.close()
    cleanup_distributed()


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--val_ratio", type=float, default=0.01)
    p.add_argument("--out_dir", type=str, default="outputs/train")
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="cosine", choices=["cosine", "linear"])
    p.add_argument("--cfg_drop_prob", type=float, default=0.1)
    p.add_argument("--ema_decay", type=float, default=0.99995)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_videos", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--shape_check", action="store_true")
    p.add_argument("--overfit_16", action="store_true")
    p.add_argument("--vis_every", type=int, default=5)
    p.add_argument("--vis_guidance_scale", type=float, default=1.5)
    p.add_argument("--vis_steps", type=int, default=50)
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--log_dir", type=str, default="runs")
    p.add_argument("--eval_fvd_every", type=int, default=0)
    p.add_argument("--num_eval_videos", type=int, default=32)
    p.add_argument("--cache_videos", action="store_true")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parse_with_config(parser)
    train(args)
