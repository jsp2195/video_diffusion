import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(channels: int) -> nn.GroupNorm:
    groups = 32 if channels >= 32 else 1
    return nn.GroupNorm(groups, channels)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temporal_kernel: int = 3):
        super().__init__()
        pad_t = temporal_kernel // 2
        self.in_layers = nn.Sequential(
            norm(in_ch),
            nn.SiLU(),
            nn.Conv3d(in_ch, out_ch, kernel_size=(temporal_kernel, 3, 3), padding=(pad_t, 1, 1)),
        )
        self.out_layers = nn.Sequential(
            norm(out_ch),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=(temporal_kernel, 3, 3), padding=(pad_t, 1, 1)),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.out_layers(self.in_layers(x))


class Downsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        return self.conv(x)
