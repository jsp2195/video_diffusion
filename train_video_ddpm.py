import argparse
import os
import random
from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.kinetics_video_dataset import KineticsVideoDataset, discover_and_split_videos
from diffusion.schedule import DiffusionSchedule
from metrics.fvd import compute_fvd_proxy
from models.video_unet3d import VideoUNet3DConditional
from utils.config import parse_with_config
from utils.diagnostics import save_training_curves
from utils.distributed import cleanup_distributed, init_distributed, is_main_process
from utils.ema import EMA
from utils.io import save_mp4
from utils.logger import TrainLogger
from tqdm.auto import tqdm

MODEL_TYPE = "video_unet3d_cond"
PREDICTION_TARGET = "v"
TASK_MODE = "endpoint_conditioned_middle_generation"


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


def build_model_config(args: argparse.Namespace) -> Dict:
    return {
        "model_type": MODEL_TYPE,
        "in_channels": 1,
        "cond_channels": 1,
        "base_channels": args.base_channels,
        "channel_mult": list(args.channel_mults),
        "num_res_blocks": args.res_blocks,
        "temporal_attn_levels": list(args.temporal_attn_levels),
        "cond_encoder_type": "multiscale_pyramid_shared",
        "endpoint_fusion_mode": "concat_proj",
        "cond_injection_mode": "film",
        "prediction_target": PREDICTION_TARGET,
        "task_mode": TASK_MODE,
        "T": args.T,
        "frame_stride": args.frame_stride,
    }


def make_model_from_config(model_cfg: Dict) -> VideoUNet3DConditional:
    if model_cfg.get("model_type") != MODEL_TYPE:
        raise ValueError(f"Unsupported model_type={model_cfg.get('model_type')}")
    if model_cfg.get("prediction_target", PREDICTION_TARGET) != PREDICTION_TARGET:
        raise ValueError("Only v-prediction checkpoints are supported in this baseline.")
    if model_cfg.get("task_mode", TASK_MODE) != TASK_MODE:
        raise ValueError(f"Unsupported task_mode={model_cfg.get('task_mode')}. Expected {TASK_MODE}.")
    if model_cfg.get("cond_encoder_type", "multiscale_pyramid_shared") != "multiscale_pyramid_shared":
        raise ValueError("Unsupported cond_encoder_type. Expected multiscale_pyramid_shared.")
    if model_cfg.get("endpoint_fusion_mode", "concat_proj") != "concat_proj":
        raise ValueError("Unsupported endpoint_fusion_mode. Expected concat_proj.")
    if model_cfg.get("cond_injection_mode", "film") != "film":
        raise ValueError("Unsupported cond_injection_mode. Expected film.")

    return VideoUNet3DConditional(
        in_channels=model_cfg.get("in_channels", 1),
        cond_channels=model_cfg.get("cond_channels", 1),
        base_channels=model_cfg["base_channels"],
        channel_mult=tuple(model_cfg["channel_mult"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        temporal_attn_levels=tuple(model_cfg.get("temporal_attn_levels", [1, 2])),
        cond_injection_mode=model_cfg.get("cond_injection_mode", "film"),
        endpoint_fusion_mode=model_cfg.get("endpoint_fusion_mode", "concat_proj"),
    )


@torch.no_grad()
def sample_video(model, diffusion, cond_start, cond_end, middle_len: int, size: int, steps: int, guidance_scale: float, eta: float = 0.0, dynamic_threshold: bool = False):
    device = cond_start.device
    x = torch.randn(1, 1, middle_len, size, size, device=device)
    ddim_times = torch.linspace(diffusion.timesteps - 1, 0, steps, device=device).long()
    for i in range(len(ddim_times)):
        t = ddim_times[i].view(1)
        t_prev = ddim_times[i + 1].view(1) if i < len(ddim_times) - 1 else torch.tensor([-1], device=device)
        zeros_start = torch.zeros_like(cond_start)
        zeros_end = torch.zeros_like(cond_end)
        v_u = model(x, t, zeros_start, zeros_end)
        v_c = model(x, t, cond_start, cond_end)
        v = v_u + guidance_scale * (v_c - v_u)
        x = diffusion.ddim_step_from_v(x, v, t, t_prev, eta=eta, dynamic_threshold=dynamic_threshold)
    return x


@torch.no_grad()
def evaluate_fvd_proxy(ema_model, diffusion, val_ds, args, device):
    num_eval = min(args.num_eval_videos, len(val_ds))
    real_videos = []
    gen_videos = []
    rng = random.Random(args.seed)
    indices = [rng.randrange(len(val_ds)) for _ in range(num_eval)]

    for idx in indices:
        sample = val_ds[idx]
        clip = sample["clip"].unsqueeze(0).to(device)
        cond_start = clip[:, :, 0]
        cond_end = clip[:, :, -1]

        gen_middle = sample_video(
            ema_model,
            diffusion,
            cond_start,
            cond_end,
            middle_len=args.T - 2,
            size=args.size,
            steps=args.vis_steps,
            guidance_scale=args.vis_guidance_scale,
            eta=0.0,
            dynamic_threshold=args.dynamic_threshold,
        )
        gen = torch.cat([cond_start.unsqueeze(2), gen_middle, cond_end.unsqueeze(2)], dim=2)
        real_videos.append(clip)
        gen_videos.append(gen)

    real_batch = torch.cat(real_videos, dim=0)
    gen_batch = torch.cat(gen_videos, dim=0)
    return compute_fvd_proxy(real_batch, gen_batch, device)


def train(args):
    if args.T < 3:
        raise ValueError("--T must be >= 3 for endpoint-conditioned middle-frame generation.")

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

    train_ds = KineticsVideoDataset(train_files, num_frames=args.T, frame_stride=args.frame_stride, size=args.size, cache_videos=args.cache_videos)
    val_ds = KineticsVideoDataset(val_files, num_frames=args.T, frame_stride=args.frame_stride, size=args.size, cache_videos=args.cache_videos)

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

    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty. Dataset too small or batch_size too large.")
    if len(val_loader) == 0:
        raise RuntimeError("Validation loader is empty.")

    model_cfg = build_model_config(args)
    model = make_model_from_config(model_cfg).to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    ema = EMA(unwrap_model(model), decay=args.ema_decay, update_after_step=args.ema_update_after_step)
    diffusion = DiffusionSchedule(args.timesteps, schedule=args.beta_schedule).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(train_loader) * args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    logger = TrainLogger(enabled=args.tensorboard and is_main_process(), log_dir=args.log_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    resume_ckpt = os.path.join(args.out_dir, "last.pt")
    if args.resume and os.path.exists(resume_ckpt):
        if is_main_process():
            print(f"Resuming training from {resume_ckpt}")

        ckpt = torch.load(resume_ckpt, map_location=device)
        ckpt_model_cfg = ckpt.get("model_config", {})
        if ckpt_model_cfg and ckpt_model_cfg != model_cfg:
            if is_main_process():
                print("[warning] CLI model args differ from checkpoint model_config. Using checkpoint config for safety.")
            model_cfg = ckpt_model_cfg
            new_model = make_model_from_config(model_cfg).to(device)
            if distributed:
                new_model = DDP(
                    new_model,
                    device_ids=[local_rank] if torch.cuda.is_available() else None,
                    output_device=local_rank if torch.cuda.is_available() else None,
                    find_unused_parameters=False,
                )
            model = new_model
            ema = EMA(unwrap_model(model), decay=args.ema_decay, update_after_step=args.ema_update_after_step)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(train_loader) * args.epochs))

        unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        ema.load_state_dict(ckpt["ema_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        sched.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        global_step = ckpt["step"]
        start_epoch = ckpt["epoch"]
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)

    fixed_rng = random.Random(args.seed)
    fixed_idx = fixed_rng.randrange(len(val_ds)) if len(val_ds) > 0 else None

    history = {"train_loss": [], "val_loss": [], "fvd_proxy": [], "fvd_proxy_epochs": []}

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_losses = []
        pbar = tqdm(
            train_loader,
            disable=not is_main_process(),
            desc=f"epoch {epoch+1}/{args.epochs}",
            leave=False,
        )

        for batch in pbar:
            clip = batch["clip"].to(device, non_blocking=True)
            cond_start = clip[:, :, 0]
            cond_end = clip[:, :, -1]
            middle = clip[:, :, 1:-1]

            if args.cfg_drop_prob > 0:
                mask = (torch.rand(cond_start.shape[0], device=device) < args.cfg_drop_prob).float().view(-1, 1, 1, 1)
                keep = (1.0 - mask)
                cond_start = cond_start * keep
                cond_end = cond_end * keep

            t = diffusion.sample_timesteps(cond_start.shape[0], device)
            x_t, noise = diffusion.forward_noise(middle, t, noise_offset=args.noise_offset)
            v_target = diffusion.velocity_target(middle, noise, t)

            amp_ctx = torch.cuda.amp.autocast(enabled=args.amp) if torch.cuda.is_available() else nullcontext()
            with amp_ctx:
                pred_v = model(x_t, t, cond_start, cond_end)
                loss = F.mse_loss(pred_v, v_target)
                if args.temporal_loss_weight > 0:
                    pred_x0 = diffusion.predict_x0_from_v(x_t, pred_v, t)
                    pred_dt = pred_x0[:, :, 1:] - pred_x0[:, :, :-1]
                    target_dt = middle[:, :, 1:] - middle[:, :, :-1]
                    loss = loss + args.temporal_loss_weight * F.l1_loss(pred_dt, target_dt)

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
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if global_step % args.log_every == 0 and is_main_process():
                logger.add_scalar("train/loss", float(loss.item()), global_step)
                logger.add_scalar("learning_rate", float(opt.param_groups[0]["lr"]), global_step)

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        train_loss_epoch = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                clip = batch["clip"].to(device, non_blocking=True)
                cond_start = clip[:, :, 0]
                cond_end = clip[:, :, -1]
                middle = clip[:, :, 1:-1]
                t = diffusion.sample_timesteps(cond_start.shape[0], device)
                x_t, noise = diffusion.forward_noise(middle, t, noise_offset=args.noise_offset)
                v_target = diffusion.velocity_target(middle, noise, t)
                pred_v = model(x_t, t, cond_start, cond_end)
                val_loss = F.mse_loss(pred_v, v_target)
                if args.temporal_loss_weight > 0:
                    pred_x0 = diffusion.predict_x0_from_v(x_t, pred_v, t)
                    pred_dt = pred_x0[:, :, 1:] - pred_x0[:, :, :-1]
                    target_dt = middle[:, :, 1:] - middle[:, :, :-1]
                    val_loss = val_loss + args.temporal_loss_weight * F.l1_loss(pred_dt, target_dt)
                val_losses.append(val_loss.item())

        val_loss_epoch = float(np.mean(val_losses)) if val_losses else 0.0

        if distributed:
            t_train = torch.tensor([train_loss_epoch], device=device)
            t_val = torch.tensor([val_loss_epoch], device=device)
            dist.all_reduce(t_train, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_val, op=dist.ReduceOp.SUM)
            train_loss_epoch = (t_train / world_size).item()
            val_loss_epoch = (t_val / world_size).item()

        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_loss_epoch)

        if is_main_process():
            print(f"epoch={epoch+1} train_loss={train_loss_epoch:.6f} val_loss={val_loss_epoch:.6f}")
            ckpt = {
                "model_state_dict": unwrap_model(model).state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "step": global_step,
                "epoch": epoch + 1,
                "best_val_loss": min(best_val_loss, val_loss_epoch),
                "config": vars(args),
                "model_config": model_cfg,
            }
            torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))

            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))

            logger.add_scalar("val/loss", val_loss_epoch, epoch + 1)

            ema_model = ema.ema_model
            if (epoch + 1) % args.vis_every == 0 and fixed_idx is not None:
                preview_sample = val_ds[fixed_idx]
                preview_clip = preview_sample["clip"].unsqueeze(0).to(device)
                preview_start = preview_clip[:, :, 0]
                preview_end = preview_clip[:, :, -1]
                preview_middle = sample_video(
                    ema_model,
                    diffusion,
                    preview_start,
                    preview_end,
                    middle_len=args.T - 2,
                    size=args.size,
                    steps=args.vis_steps,
                    guidance_scale=args.vis_guidance_scale,
                    eta=0.0,
                    dynamic_threshold=args.dynamic_threshold,
                )
                preview_video = torch.cat([preview_start.unsqueeze(2), preview_middle, preview_end.unsqueeze(2)], dim=2)
                preview_path = os.path.join(args.out_dir, f"epoch_{epoch+1:04d}_sample.mp4")
                save_mp4(preview_video[0], preview_path, fps=8)
                logger.add_video("preview/video", ((preview_video.clamp(-1, 1) + 1.0) / 2.0).permute(0, 2, 1, 3, 4), epoch + 1, fps=8)

            if args.eval_fvd_every > 0 and (epoch + 1) % args.eval_fvd_every == 0:
                fvd_proxy_score = evaluate_fvd_proxy(ema_model, diffusion, val_ds, args, device)
                history["fvd_proxy"].append(fvd_proxy_score)
                history["fvd_proxy_epochs"].append(epoch + 1)
                print(f"epoch={epoch+1}, fvd_proxy={fvd_proxy_score:.4f}")
                logger.add_scalar("metrics/fvd_proxy", fvd_proxy_score, epoch + 1)

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
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="cosine", choices=["cosine", "linear"])
    p.add_argument("--cfg_drop_prob", type=float, default=0.08)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--ema_update_after_step", type=int, default=100)
    p.add_argument("--base_channels", type=int, default=96)
    p.add_argument("--channel_mults", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--res_blocks", type=int, default=2)
    p.add_argument("--temporal_attn_levels", type=int, nargs="+", default=[1, 2])
    p.add_argument("--temporal_loss_weight", type=float, default=0.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_videos", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overfit_16", action="store_true")
    p.add_argument("--vis_every", type=int, default=1)
    p.add_argument("--vis_guidance_scale", type=float, default=1.5)
    p.add_argument("--vis_steps", type=int, default=40)
    p.add_argument("--noise_offset", type=float, default=0.0)
    p.add_argument("--dynamic_threshold", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--log_dir", type=str, default="runs")
    p.add_argument("--eval_fvd_every", type=int, default=0)
    p.add_argument("--num_eval_videos", type=int, default=32)
    p.add_argument("--cache_videos", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parse_with_config(parser)
    train(args)
