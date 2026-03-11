import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.conditioning_encoder import ConditioningEncoder
from models.factorized_attention import FactorizedSpaceTimeBlock
from models.patch_embed3d import PatchDecode3D, PatchEmbed3D


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(self.max_positions) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class VideoDiffusionTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 128,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_heads: int = 8,
        max_temporal_length: int = 32,
        embed_dim: int = 512,
        depth: int = 12,
        patch_size: tuple[int, int, int] = (2, 4, 4),
        max_spatial_tokens: int = 24 * 24,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        _ = (base_channels, channel_mult, num_res_blocks)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint

        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size=patch_size)
        self.patch_decode = PatchDecode3D(embed_dim, in_channels, patch_size=patch_size)

        cond_channels = embed_dim
        self.cond_encoder = ConditioningEncoder(in_channels=in_channels, out_channels=cond_channels)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.temporal_pos = nn.Parameter(torch.randn(max_temporal_length, embed_dim) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(max_spatial_tokens, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([FactorizedSpaceTimeBlock(embed_dim, heads=attention_heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(embed_dim)

    def _add_positional_embeddings(self, x: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        b, _, c = x.shape
        tp, hp, wp = grid
        s = hp * wp
        t_pos = self.temporal_pos[:tp].view(1, tp, 1, c)
        s_pos = self.spatial_pos[:s].view(1, 1, s, c)
        pos = (t_pos + s_pos).reshape(1, tp * s, c)
        return x + pos.expand(b, -1, -1)

    def _cond_tokens(self, cond_first_frame: torch.Tensor) -> torch.Tensor:
        cond_feats = self.cond_encoder(cond_first_frame)
        b, c, h, w = cond_feats.shape
        return cond_feats.view(b, c, h * w).transpose(1, 2)

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor, cond_first_frame: torch.Tensor) -> torch.Tensor:
        x, grid = self.patch_embed(x_t)
        x = self._add_positional_embeddings(x, grid)
        t_emb = self.time_mlp(timestep)
        cond_tokens = self._cond_tokens(cond_first_frame)

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(lambda _x, _c, _t: block(_x, _c, _t, grid), x, cond_tokens, t_emb, use_reentrant=False)
            else:
                x = block(x, cond_tokens, t_emb, grid)

        x = self.final_norm(x)
        return self.patch_decode(x, grid)
