import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    # extracted from original DDPM_pytorch_efv.py
    def __init__(self, dim: int, max_positions: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        half_dim = self.dim // 2
        emb = math.log(self.max_positions) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch * 2))
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        gamma, beta = self.time_mlp(t_emb).chunk(2, dim=-1)
        h = h * (1 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]
        h = F.silu(h)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class ResBlock2D(nn.Module):
    # adapted from 2D control blocks in original file
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class ControlEncoder2D(nn.Module):
    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock2D(1, 64),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            ResBlock2D(64, 128),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            ResBlock2D(128, out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VideoUNetConditional(nn.Module):
    def __init__(self, in_channels: int = 1, base_dim: int = 64, cond_dim: int = 256):
        super().__init__()
        self.cond_encoder = ControlEncoder2D(out_channels=cond_dim)
        self.time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.in_proj = nn.Conv3d(in_channels + cond_dim, base_dim, 3, padding=1)

        self.down1 = ResBlock3D(base_dim + cond_dim, base_dim, self.time_dim)
        self.down2 = ResBlock3D(base_dim * 2 + cond_dim, base_dim * 2, self.time_dim)
        self.down3 = ResBlock3D(base_dim * 4 + cond_dim, base_dim * 4, self.time_dim)

        self.ds = nn.Conv3d(base_dim, base_dim * 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.ds2 = nn.Conv3d(base_dim * 2, base_dim * 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

        self.mid = ResBlock3D(base_dim * 4 + cond_dim, base_dim * 4, self.time_dim)

        self.us1 = nn.ConvTranspose3d(base_dim * 4, base_dim * 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.up1 = ResBlock3D(base_dim * 4 + cond_dim, base_dim * 2, self.time_dim)

        self.us2 = nn.ConvTranspose3d(base_dim * 2, base_dim, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.up2 = ResBlock3D(base_dim * 2 + cond_dim, base_dim, self.time_dim)

        self.out = nn.Conv3d(base_dim, in_channels, 1)

    def _cond_3d(self, cond_first_frame: torch.Tensor, t_len: int, hw: Tuple[int, int]) -> torch.Tensor:
        cond2d = self.cond_encoder(cond_first_frame)
        cond2d = F.interpolate(cond2d, size=hw, mode="bilinear", align_corners=False)
        return cond2d.unsqueeze(2).repeat(1, 1, t_len, 1, 1)

    def forward(self, noisy_clip: torch.Tensor, t: torch.Tensor, cond_first_frame: torch.Tensor) -> torch.Tensor:
        _, _, t_len, h, w = noisy_clip.shape
        t_emb = self.time_mlp(t)

        c0 = self._cond_3d(cond_first_frame, t_len, (h, w))
        x = self.in_proj(torch.cat([noisy_clip, c0], dim=1))

        c1 = self._cond_3d(cond_first_frame, t_len, (x.shape[-2], x.shape[-1]))
        x1 = self.down1(torch.cat([x, c1], dim=1), t_emb)

        x2_in = self.ds(x1)
        c2 = self._cond_3d(cond_first_frame, t_len, (x2_in.shape[-2], x2_in.shape[-1]))
        x2 = self.down2(torch.cat([x2_in, c2], dim=1), t_emb)

        x3_in = self.ds2(x2)
        c3 = self._cond_3d(cond_first_frame, t_len, (x3_in.shape[-2], x3_in.shape[-1]))
        x3 = self.down3(torch.cat([x3_in, c3], dim=1), t_emb)

        cmid = self._cond_3d(cond_first_frame, t_len, (x3.shape[-2], x3.shape[-1]))
        xm = self.mid(torch.cat([x3, cmid], dim=1), t_emb)

        u1 = self.us1(xm)
        u1 = torch.cat([u1, x2], dim=1)
        cu1 = self._cond_3d(cond_first_frame, t_len, (u1.shape[-2], u1.shape[-1]))
        u1 = self.up1(torch.cat([u1, cu1], dim=1), t_emb)

        u2 = self.us2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        cu2 = self._cond_3d(cond_first_frame, t_len, (u2.shape[-2], u2.shape[-1]))
        u2 = self.up2(torch.cat([u2, cu2], dim=1), t_emb)

        return self.out(u2)
