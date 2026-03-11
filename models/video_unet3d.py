import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conditioning_encoder import ConditioningEncoder


def _group_count(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


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


class EndpointFeatureFusion(nn.Module):
    """Fuse start/end 2D endpoint features into a single conditional map."""

    def __init__(self, feat_ch: int, out_ch: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_ch * 2, out_ch, kernel_size=1),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, start_feat: torch.Tensor, end_feat: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([start_feat, end_feat], dim=1))


class FiLMInjection(nn.Module):
    """FiLM-like conditioning injection from 2D endpoint feature into 3D feature maps."""

    def __init__(self, cond_ch: int, target_ch: int):
        super().__init__()
        self.to_scale = nn.Conv2d(cond_ch, target_ch, kernel_size=1)
        self.to_shift = nn.Conv2d(cond_ch, target_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, cond_feat_2d: torch.Tensor) -> torch.Tensor:
        scale = self.to_scale(cond_feat_2d).unsqueeze(2)
        shift = self.to_shift(cond_feat_2d).unsqueeze(2)
        return x * (1.0 + scale) + shift


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_channels)
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb).view(t_emb.shape[0], -1, 1, 1, 1)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class TemporalSelfAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        h_in = x
        x = self.norm(x).permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x.reshape(b, h, w, t, c).permute(0, 4, 3, 1, 2).contiguous()
        return self.proj(x) + h_in


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, num_res_blocks: int, with_temporal_attn: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = in_ch
        for _ in range(num_res_blocks):
            self.blocks.append(ResBlock3D(ch, out_ch, time_dim))
            ch = out_ch
        self.temporal_attn = TemporalSelfAttention(out_ch) if with_temporal_attn else nn.Identity()
        self.downsample = nn.Conv3d(out_ch, out_ch, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.temporal_attn(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int, num_res_blocks: int, with_temporal_attn: bool = False):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.blocks = nn.ModuleList()
        ch = out_ch + skip_ch
        for _ in range(num_res_blocks):
            self.blocks.append(ResBlock3D(ch, out_ch, time_dim))
            ch = out_ch
        self.temporal_attn = TemporalSelfAttention(out_ch) if with_temporal_attn else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=(skip.shape[2], skip.shape[3], skip.shape[4]), mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = block(x, t_emb)
        return self.temporal_attn(x)


class VideoUNet3DConditional(nn.Module):
    """Compact endpoint-conditioned 3D U-Net for middle-frame denoising."""

    def __init__(
        self,
        in_channels: int = 1,
        cond_channels: int = 1,
        base_channels: int = 96,
        channel_mult: Iterable[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        temporal_attn_levels: Iterable[int] = (1, 2),
        cond_injection_mode: str = "film",
        endpoint_fusion_mode: str = "concat_proj",
        endpoint_context: int = 2,
    ):
        super().__init__()
        if cond_injection_mode != "film":
            raise ValueError("Only cond_injection_mode='film' is supported.")
        if endpoint_fusion_mode != "concat_proj":
            raise ValueError("Only endpoint_fusion_mode='concat_proj' is supported.")

        channel_mult = list(channel_mult)
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.endpoint_context = endpoint_context
        self.temporal_attn_levels = tuple(temporal_attn_levels)
        self.cond_injection_mode = cond_injection_mode
        self.endpoint_fusion_mode = endpoint_fusion_mode

        self.cond_encoder = ConditioningEncoder(
            in_channels=cond_channels * endpoint_context,
            base_channels=base_channels,
            channel_mult=channel_mult,
        )

        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        input_ch = in_channels + cond_channels * 2
        self.in_conv = nn.Conv3d(input_ch, base_channels, 3, padding=1)

        level_channels: List[int] = [base_channels * mult for mult in channel_mult]

        self.stem_fuse = EndpointFeatureFusion(level_channels[0], level_channels[0])
        self.stem_inject = FiLMInjection(level_channels[0], base_channels)

        self.down_blocks = nn.ModuleList()
        self.down_fuse = nn.ModuleList()
        self.down_injects = nn.ModuleList()
        self.skip_injects = nn.ModuleList()

        prev_ch = base_channels
        attn_levels_set = set(temporal_attn_levels)
        for level_idx, out_ch in enumerate(level_channels):
            use_attn = level_idx in attn_levels_set
            self.down_fuse.append(EndpointFeatureFusion(level_channels[level_idx], level_channels[level_idx]))
            self.down_injects.append(FiLMInjection(level_channels[level_idx], prev_ch))
            self.down_blocks.append(DownBlock(prev_ch, out_ch, time_dim, num_res_blocks, with_temporal_attn=use_attn))
            self.skip_injects.append(FiLMInjection(level_channels[level_idx], out_ch))
            prev_ch = out_ch

        self.mid_fuse = EndpointFeatureFusion(level_channels[-1], level_channels[-1])
        self.mid_inject = FiLMInjection(level_channels[-1], prev_ch)
        self.mid1 = ResBlock3D(prev_ch, prev_ch, time_dim)
        self.mid_attn = TemporalSelfAttention(prev_ch, heads=4)
        self.mid2 = ResBlock3D(prev_ch, prev_ch, time_dim)

        self.up_blocks = nn.ModuleList()
        self.up_injects = nn.ModuleList()
        for level_idx in reversed(range(len(level_channels))):
            skip_ch = level_channels[level_idx]
            out_ch = skip_ch
            use_attn = level_idx in attn_levels_set
            self.up_blocks.append(UpBlock(prev_ch, skip_ch, out_ch, time_dim, num_res_blocks, with_temporal_attn=use_attn))
            self.up_injects.append(FiLMInjection(level_channels[level_idx], out_ch))
            prev_ch = out_ch

        self.out = nn.Sequential(
            nn.GroupNorm(_group_count(prev_ch), prev_ch),
            nn.SiLU(),
            nn.Conv3d(prev_ch, in_channels, 3, padding=1),
        )


    def _encode_context(self, context_clip: torch.Tensor) -> dict:
        # [B,C,K,H,W] -> [B,C*K,H,W]
        b, c, k, h, w = context_clip.shape
        if k != self.endpoint_context:
            raise ValueError(f"Expected endpoint_context={self.endpoint_context}, got {k}")
        return self.cond_encoder(context_clip.reshape(b, c * k, h, w))

    def _fuse_endpoint_pyramids(self, start_context: torch.Tensor, end_context: torch.Tensor):
        start_feats = self._encode_context(start_context)
        end_feats = self._encode_context(end_context)

        fused_stem = self.stem_fuse(start_feats["stem"], end_feats["stem"])
        fused_down = []
        for i, fuse in enumerate(self.down_fuse):
            fused_down.append(fuse(start_feats["down"][i], end_feats["down"][i]))
        fused_mid = self.mid_fuse(start_feats["mid"], end_feats["mid"])
        return fused_stem, fused_down, fused_mid

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor, cond_start_context: torch.Tensor, cond_end_context: torch.Tensor) -> torch.Tensor:
        start_key = cond_start_context[:, :, -1]
        end_key = cond_end_context[:, :, 0]
        start_video = start_key.unsqueeze(2).repeat(1, 1, x_t.shape[2], 1, 1)
        end_video = end_key.unsqueeze(2).repeat(1, 1, x_t.shape[2], 1, 1)
        fused_stem, fused_down, fused_mid = self._fuse_endpoint_pyramids(cond_start_context, cond_end_context)

        x = torch.cat([x_t, start_video, end_video], dim=1)
        t_emb = self.time_mlp(timestep)
        x = self.in_conv(x)
        x = self.stem_inject(x, fused_stem)

        skips = []
        for level_idx, down in enumerate(self.down_blocks):
            x = self.down_injects[level_idx](x, fused_down[level_idx])
            x, skip = down(x, t_emb)
            skip = self.skip_injects[level_idx](skip, fused_down[level_idx])
            skips.append(skip)

        x = self.mid_inject(x, fused_mid)
        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        for i, up in enumerate(self.up_blocks):
            level_idx = len(self.up_blocks) - 1 - i
            x = up(x, skips.pop(), t_emb)
            x = self.up_injects[i](x, fused_down[level_idx])

        return self.out(x)
