import torch
import torch.nn as nn

from models.conditioning.image_embedder_gray import GrayImageEmbedder
from models.conditioning.latent_conditioner import expand_cond_latent


class GeneralConditioner(nn.Module):
    def __init__(self, ae, context_dim: int = 512, fps_bins: int = 32, motion_bins: int = 32):
        super().__init__()
        self.ae = ae
        self.image_embedder = GrayImageEmbedder(in_channels=1, embed_dim=context_dim)
        self.fps_embed = nn.Embedding(fps_bins, context_dim)
        self.motion_embed = nn.Embedding(motion_bins, context_dim)
        self.cond_aug_mlp = nn.Sequential(nn.Linear(1, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim))

    @torch.no_grad()
    def encode_cond_latent(self, cond_frame: torch.Tensor) -> torch.Tensor:
        # cond_frame [B,1,H,W]
        x = cond_frame.unsqueeze(2)
        mean, _ = self.ae.encode(x)
        return mean[:, :, 0]

    def forward(self, batch: dict, num_frames: int):
        cond_frame = batch["cond_frames"]
        cond_latent = self.encode_cond_latent(cond_frame)
        concat_cond = expand_cond_latent(cond_latent, num_frames)

        ctx_tokens = self.image_embedder(cond_frame)
        fps = batch["fps_id"].long().clamp(min=0, max=self.fps_embed.num_embeddings - 1)
        motion = batch["motion_bucket_id"].long().clamp(min=0, max=self.motion_embed.num_embeddings - 1)
        cond_aug = batch["cond_aug"].view(-1, 1)

        vec = self.fps_embed(fps) + self.motion_embed(motion) + self.cond_aug_mlp(cond_aug)
        return {"concat": concat_cond, "context": ctx_tokens, "vector": vec}
