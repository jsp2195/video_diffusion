import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.kinetics_video_dataset import KineticsVideoDataset, discover_and_split_videos
from models.autoencoder.autoencoder_kl_gray import AutoencoderKLGray
from utils.configuration import load_config, parse_args


def main():
    args = parse_args("configs/training/ae_stage.yaml")
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files, _, _, _ = discover_and_split_videos(cfg["data"]["root"], cfg["data"]["val_ratio"], cfg["seed"], cfg["data"].get("max_videos"))
    ds = KineticsVideoDataset(train_files, num_frames=cfg["data"]["num_frames"], size=cfg["data"]["size"], cache_videos=cfg["data"].get("cache_videos", False))
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"], drop_last=True)

    model = AutoencoderKLGray(**cfg["first_stage"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"].get("amp", True))

    os.makedirs(cfg["training"]["out_dir"], exist_ok=True)
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        pbar = tqdm(loader, desc=f"ae epoch {epoch+1}")
        for batch in pbar:
            video = batch["video"].to(device)
            with torch.cuda.amp.autocast(enabled=cfg["training"].get("amp", True)):
                recon, mean, logvar = model(video)
                recon_loss = F.l1_loss(recon, video)
                kl = model.kl_loss(mean, logvar)
                loss = recon_loss + cfg["training"].get("kl_weight", 1e-4) * kl
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon_loss.item():.4f}")

        ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch}
        torch.save(ckpt, os.path.join(cfg["training"]["out_dir"], "ae_last.pt"))


if __name__ == "__main__":
    main()
