import math

import torch
import torch.nn as nn

from models.attention import SpatialAttention
from models.positional_encoding import FramePositionalEncoding
from models.resblocks import VideoResBlock
from models.temporal_modules import TemporalConvBlock


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        half = self.dim // 2
        emb = math.log(self.max_positions) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class VideoStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.block1 = VideoResBlock(in_ch, out_ch, time_dim)
        self.s_attn = SpatialAttention(out_ch, num_heads=4)
        self.frame_pe = FramePositionalEncoding(out_ch)
        self.t_conv = TemporalConvBlock(out_ch)
        self.block2 = VideoResBlock(out_ch, out_ch, time_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.block1(x, t_emb)
        x = self.s_attn(x)
        x = self.frame_pe(x)
        x = self.t_conv(x)
        x = self.block2(x, t_emb)
        return x


class VideoUNetConditional(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 96):
        super().__init__()
        time_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # x_t + conditioning frame
        self.in_conv = nn.Conv3d(in_channels + 1, base_channels, 3, padding=1)

        # Encoder
        self.down1 = VideoStage(base_channels, base_channels, time_dim)
        self.down2 = VideoStage(base_channels * 2, base_channels * 2, time_dim)
        self.down3 = VideoStage(base_channels * 4, base_channels * 4, time_dim)
        self.down4 = VideoStage(base_channels * 8, base_channels * 8, time_dim)

        self.ds1 = nn.Conv3d(base_channels, base_channels * 2, (1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.ds2 = nn.Conv3d(base_channels * 2, base_channels * 4, (1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.ds3 = nn.Conv3d(base_channels * 4, base_channels * 8, (1,4,4), stride=(1,2,2), padding=(0,1,1))

        # Bottleneck
        self.mid = VideoStage(base_channels * 8, base_channels * 8, time_dim)

        # Decoder
        self.us1 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, (1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.up1 = VideoStage(base_channels * 8, base_channels * 4, time_dim)

        self.us2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, (1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.up2 = VideoStage(base_channels * 4, base_channels * 2, time_dim)

        self.us3 = nn.ConvTranspose3d(base_channels * 2, base_channels, (1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.up3 = VideoStage(base_channels * 2, base_channels, time_dim)

        self.out_norm = nn.GroupNorm(32, base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv3d(base_channels, in_channels, 3, padding=1)

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor, cond_first_frame: torch.Tensor) -> torch.Tensor:
        b, _, t, h, w = x_t.shape
        t_emb = self.time_mlp(timestep)

        cond_rep = cond_first_frame.unsqueeze(2).repeat(1, 1, t, 1, 1)
        x = torch.cat([x_t, cond_rep], dim=1)
        x = self.in_conv(x)

        s1 = self.down1(x, t_emb)
        x = self.ds1(s1)

        s2 = self.down2(x, t_emb)
        x = self.ds2(s2)

        s3 = self.down3(x, t_emb)
        x = self.ds3(s3)

        x = self.down4(x, t_emb)
        x = self.mid(x, t_emb)

        x = self.us1(x)
        x = torch.cat([x, s3], dim=1)
        x = self.up1(x, t_emb)

        x = self.us2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.up2(x, t_emb)

        x = self.us3(x)
        x = torch.cat([x, s1], dim=1)
        x = self.up3(x, t_emb)

        x = self.out_conv(self.out_act(self.out_norm(x)))
        return x
