import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_groups(channels: int, max_groups: int = 32):
    g = min(max_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g


class VideoResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()

        g1 = _valid_groups(in_channels)
        g2 = _valid_groups(out_channels)

        self.norm1 = nn.GroupNorm(g1, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(g2, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        t = self.time_proj(t_emb)
        h = h + t[:, :, None, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)
