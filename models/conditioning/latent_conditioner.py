import torch


def expand_cond_latent(cond_latent: torch.Tensor, t: int) -> torch.Tensor:
    # cond_latent [B,C,H,W] -> [B,C,T,H,W]
    return cond_latent.unsqueeze(2).repeat(1, 1, t, 1, 1)
