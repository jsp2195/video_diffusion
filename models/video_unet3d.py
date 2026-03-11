import math

import torch
import torch.nn as nn

from models.conditioning_encoder import ConditioningEncoder
from models.temporal_attention import TemporalAttention


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


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, max_temporal_length: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act2 = nn.SiLU()

        self.time_proj = nn.Linear(time_dim, out_channels)
        self.temporal_pos = nn.Parameter(torch.randn(max_temporal_length, out_channels) * 0.02)

        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        b, _, t, _, _ = x.shape
        if t > self.temporal_pos.shape[0]:
            raise ValueError(f"Input T={t} exceeds max temporal length={self.temporal_pos.shape[0]}")

        res = self.skip(x)

        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        temb = self.time_proj(t_emb).view(b, -1, 1, 1, 1)
        tpos = self.temporal_pos[:t].transpose(0, 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        h = h + temb + tpos

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)

        return h + res


class VideoUNet3DConditional(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 128,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_heads: int = 8,
        max_temporal_length: int = 32,
    ):
        super().__init__()
        assert num_res_blocks == 2, "This implementation expects num_res_blocks=2."

        ch0, ch1, ch2, ch3 = [base_channels * m for m in channel_mult]
        time_dim = base_channels * 4
        cond_channels = ch3

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.cond_encoder = ConditioningEncoder(in_channels=in_channels, out_channels=cond_channels)

        self.in_conv = nn.Conv3d(in_channels, ch0, 3, padding=1)

        self.enc0a = ResBlock3D(ch0, ch0, time_dim, max_temporal_length)
        self.attn0a = TemporalAttention(ch0, cond_channels, heads=attention_heads)
        self.enc0b = ResBlock3D(ch0, ch0, time_dim, max_temporal_length)
        self.attn0b = TemporalAttention(ch0, cond_channels, heads=attention_heads)

        self.down0 = nn.Conv3d(ch0, ch1, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

        self.enc1a = ResBlock3D(ch1, ch1, time_dim, max_temporal_length)
        self.attn1a = TemporalAttention(ch1, cond_channels, heads=attention_heads)
        self.enc1b = ResBlock3D(ch1, ch1, time_dim, max_temporal_length)
        self.attn1b = TemporalAttention(ch1, cond_channels, heads=attention_heads)

        self.down1 = nn.Conv3d(ch1, ch2, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

        self.enc2a = ResBlock3D(ch2, ch2, time_dim, max_temporal_length)
        self.attn2a = TemporalAttention(ch2, cond_channels, heads=attention_heads)
        self.enc2b = ResBlock3D(ch2, ch2, time_dim, max_temporal_length)
        self.attn2b = TemporalAttention(ch2, cond_channels, heads=attention_heads)

        self.down2 = nn.Conv3d(ch2, ch3, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

        self.enc3a = ResBlock3D(ch3, ch3, time_dim, max_temporal_length)
        self.attn3a = TemporalAttention(ch3, cond_channels, heads=attention_heads)
        self.enc3b = ResBlock3D(ch3, ch3, time_dim, max_temporal_length)
        self.attn3b = TemporalAttention(ch3, cond_channels, heads=attention_heads)

        self.mid1 = ResBlock3D(ch3, ch3, time_dim, max_temporal_length)
        self.mid_attn = TemporalAttention(ch3, cond_channels, heads=attention_heads)
        self.mid2 = ResBlock3D(ch3, ch3, time_dim, max_temporal_length)

        self.dec3a = ResBlock3D(ch3 + ch3, ch3, time_dim, max_temporal_length)
        self.dec3_attn_a = TemporalAttention(ch3, cond_channels, heads=attention_heads)
        self.dec3b = ResBlock3D(ch3, ch3, time_dim, max_temporal_length)
        self.dec3_attn_b = TemporalAttention(ch3, cond_channels, heads=attention_heads)

        self.up2 = nn.ConvTranspose3d(ch3, ch2, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.dec2a = ResBlock3D(ch2 + ch2, ch2, time_dim, max_temporal_length)
        self.dec2_attn_a = TemporalAttention(ch2, cond_channels, heads=attention_heads)
        self.dec2b = ResBlock3D(ch2, ch2, time_dim, max_temporal_length)
        self.dec2_attn_b = TemporalAttention(ch2, cond_channels, heads=attention_heads)

        self.up1 = nn.ConvTranspose3d(ch2, ch1, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.dec1a = ResBlock3D(ch1 + ch1, ch1, time_dim, max_temporal_length)
        self.dec1_attn_a = TemporalAttention(ch1, cond_channels, heads=attention_heads)
        self.dec1b = ResBlock3D(ch1, ch1, time_dim, max_temporal_length)
        self.dec1_attn_b = TemporalAttention(ch1, cond_channels, heads=attention_heads)

        self.up0 = nn.ConvTranspose3d(ch1, ch0, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.dec0a = ResBlock3D(ch0 + ch0, ch0, time_dim, max_temporal_length)
        self.dec0_attn_a = TemporalAttention(ch0, cond_channels, heads=attention_heads)
        self.dec0b = ResBlock3D(ch0, ch0, time_dim, max_temporal_length)
        self.dec0_attn_b = TemporalAttention(ch0, cond_channels, heads=attention_heads)

        self.out = nn.Sequential(
            nn.GroupNorm(_group_count(ch0), ch0),
            nn.SiLU(),
            nn.Conv3d(ch0, in_channels, 3, padding=1),
        )

    def _cond_tokens(self, cond_first_frame: torch.Tensor) -> torch.Tensor:
        cond_feats = self.cond_encoder(cond_first_frame)
        b, c, h, w = cond_feats.shape
        return cond_feats.view(b, c, h * w).transpose(1, 2)

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor, cond_first_frame: torch.Tensor) -> torch.Tensor:
        cond_tokens = self._cond_tokens(cond_first_frame)
        t_emb = self.time_mlp(timestep)

        x = self.in_conv(x_t)

        x = self.attn0a(self.enc0a(x, t_emb), cond_tokens)
        x = self.attn0b(self.enc0b(x, t_emb), cond_tokens)
        s0 = x

        x = self.down0(x)
        x = self.attn1a(self.enc1a(x, t_emb), cond_tokens)
        x = self.attn1b(self.enc1b(x, t_emb), cond_tokens)
        s1 = x

        x = self.down1(x)
        x = self.attn2a(self.enc2a(x, t_emb), cond_tokens)
        x = self.attn2b(self.enc2b(x, t_emb), cond_tokens)
        s2 = x

        x = self.down2(x)
        x = self.attn3a(self.enc3a(x, t_emb), cond_tokens)
        x = self.attn3b(self.enc3b(x, t_emb), cond_tokens)
        s3 = x

        x = self.mid1(x, t_emb)
        x = self.mid_attn(x, cond_tokens)
        x = self.mid2(x, t_emb)

        x = torch.cat([x, s3], dim=1)
        x = self.dec3_attn_a(self.dec3a(x, t_emb), cond_tokens)
        x = self.dec3_attn_b(self.dec3b(x, t_emb), cond_tokens)

        x = self.up2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2_attn_a(self.dec2a(x, t_emb), cond_tokens)
        x = self.dec2_attn_b(self.dec2b(x, t_emb), cond_tokens)

        x = self.up1(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1_attn_a(self.dec1a(x, t_emb), cond_tokens)
        x = self.dec1_attn_b(self.dec1b(x, t_emb), cond_tokens)

        x = self.up0(x)
        x = torch.cat([x, s0], dim=1)
        x = self.dec0_attn_a(self.dec0a(x, t_emb), cond_tokens)
        x = self.dec0_attn_b(self.dec0b(x, t_emb), cond_tokens)

        return self.out(x)
