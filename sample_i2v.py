import argparse
import os

import numpy as np
import torch
from PIL import Image

from models.autoencoder.autoencoder_kl_gray import AutoencoderKLGray
from models.conditioning.general_conditioner import GeneralConditioner
from models.diffusion.denoiser import LatentDenoiser
from models.diffusion.samplers import euler_sample
from models.diffusion.sigma_sampling import karras_sigmas
from models.diffusion.video_unet_gray import VideoUNetGray
from utils.configuration import load_config
from utils.io import save_mp4


def load_cond(path: str, size: int):
    img = Image.open(path).convert("L").resize((size, size), resample=Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr)[None, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sampling/euler_gray_i2v.yaml")
    parser.add_argument("--cond_image", type=str, required=True)
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--ldm_ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/sample_i2v.mp4")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = AutoencoderKLGray(**cfg["first_stage"]).to(device)
    ae.load_state_dict(torch.load(args.ae_ckpt, map_location=device)["model"])
    ae.eval()

    conditioner = GeneralConditioner(ae=ae, context_dim=cfg["conditioning"]["context_dim"], motion_bins=cfg["conditioning"]["motion_bins"]).to(device)
    unet = VideoUNetGray(**cfg["diffusion_model"]).to(device)
    ldm_state = torch.load(args.ldm_ckpt, map_location=device)
    unet.load_state_dict(ldm_state["unet"])
    conditioner.load_state_dict(ldm_state["conditioner"], strict=False)
    unet.eval()
    conditioner.eval()

    cond = load_cond(args.cond_image, cfg["data"]["size"]).to(device)
    batch = {
        "cond_frames": cond,
        "fps_id": torch.tensor([cfg["sampling"]["fps_id"]], device=device),
        "motion_bucket_id": torch.tensor([cfg["sampling"]["motion_bucket_id"]], device=device),
        "cond_aug": torch.tensor([cfg["sampling"].get("cond_aug", 0.0)], device=device),
    }

    with torch.no_grad():
        cond_out = conditioner(batch, num_frames=cfg["data"]["num_frames"])
        denoiser = LatentDenoiser(unet)
        sigmas = karras_sigmas(cfg["sampling"]["steps"], cfg["sigma"]["sigma_min"], cfg["sigma"]["sigma_max"], cfg["sigma"]["rho"], device=device)
        shape = (1, cfg["first_stage"]["z_channels"], cfg["data"]["num_frames"], cfg["data"]["size"] // 8, cfg["data"]["size"] // 8)
        cond_dict = {"concat": cond_out["concat"], "context": cond_out["context"], "vector": cond_out["vector"]}
        sampled = euler_sample(denoiser, cond=cond_dict, shape=shape, sigmas=sigmas, guidance_scale=cfg["sampling"].get("guidance_scale", 1.0))
        decoded = ae.decode(sampled)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # optional final frame replace only at output time
    decoded[:, :, 0] = cond
    save_mp4(decoded, args.out, fps=cfg["sampling"].get("fps", 8))


if __name__ == "__main__":
    main()
