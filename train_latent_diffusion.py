import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.kinetics_video_dataset import KineticsVideoDataset, discover_and_split_videos
from models.autoencoder.autoencoder_kl_gray import AutoencoderKLGray
from models.conditioning.general_conditioner import GeneralConditioner
from models.diffusion.losses import v_prediction_loss, v_target
from models.diffusion.sigma_sampling import sample_sigmas
from models.diffusion.video_unet_gray import VideoUNetGray
from utils.configuration import load_config, parse_args
from utils.ema import EMA


def maybe_cfg_dropout(batch: dict, drop_prob: float):
    if drop_prob <= 0:
        return batch
    mask = (torch.rand(batch["cond_frames"].shape[0], device=batch["cond_frames"].device) < drop_prob).float()[:, None, None]
    batch = dict(batch)
    batch["cond_frames"] = batch["cond_frames"] * (1.0 - mask)
    return batch


def main():
    args = parse_args("configs/training/ldm_stage.yaml")
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files, _, _, _ = discover_and_split_videos(cfg["data"]["root"], cfg["data"]["val_ratio"], cfg["seed"], cfg["data"].get("max_videos"))
    ds = KineticsVideoDataset(train_files, num_frames=cfg["data"]["num_frames"], size=cfg["data"]["size"], cache_videos=cfg["data"].get("cache_videos", False), motion_buckets=cfg["conditioning"]["motion_bins"])
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"], drop_last=True)

    ae = AutoencoderKLGray(**cfg["first_stage"]).to(device)
    ae_ckpt = torch.load(cfg["training"]["ae_ckpt"], map_location=device)
    ae.load_state_dict(ae_ckpt["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    conditioner = GeneralConditioner(ae=ae, context_dim=cfg["conditioning"]["context_dim"], motion_bins=cfg["conditioning"]["motion_bins"]).to(device)
    unet = VideoUNetGray(**cfg["diffusion_model"]).to(device)
    params = list(unet.parameters()) + list(conditioner.image_embedder.parameters()) + list(conditioner.fps_embed.parameters()) + list(conditioner.motion_embed.parameters()) + list(conditioner.cond_aug_mlp.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["training"]["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"].get("amp", True))
    ema = EMA(unet, decay=cfg["training"].get("ema_decay", 0.999), update_after_step=0)

    os.makedirs(cfg["training"]["out_dir"], exist_ok=True)
    for epoch in range(cfg["training"]["epochs"]):
        unet.train()
        conditioner.train()
        pbar = tqdm(loader, desc=f"ldm epoch {epoch+1}")
        for batch in pbar:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch = maybe_cfg_dropout(batch, cfg["training"].get("cfg_dropout", 0.1))
            with torch.no_grad():
                mean, _ = ae.encode(batch["video"])
                latents = mean

            sigmas = sample_sigmas(latents.shape[0], device, cfg["sigma"]["sigma_min"], cfg["sigma"]["sigma_max"], cfg["sigma"]["rho"])
            noise = torch.randn_like(latents)
            noisy = latents + noise * sigmas[:, None, None, None, None]

            cond = conditioner(batch, num_frames=latents.shape[2])
            model_in = torch.cat([noisy, cond["concat"]], dim=1)

            with torch.cuda.amp.autocast(enabled=cfg["training"].get("amp", True)):
                pred_v = unet(model_in, sigmas, cond["context"], cond["vector"])
                target = v_target(latents, noisy, sigmas)
                loss = v_prediction_loss(pred_v, target, sigmas, min_snr_gamma=cfg["training"].get("min_snr_gamma", 5.0))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            ema.update(unet)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        ckpt = {
            "unet": unet.state_dict(),
            "conditioner": conditioner.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
        }
        torch.save(ckpt, os.path.join(cfg["training"]["out_dir"], "ldm_last.pt"))


if __name__ == "__main__":
    main()
