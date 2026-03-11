import torch
import torch.nn as nn

from models.autoencoder.blocks import Downsample3D, ResBlock3D, norm


class GrayVideoEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 128, channel_mult=(1, 2, 4, 4), num_res_blocks: int = 2, z_channels: int = 4):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        ch = base_channels
        blocks = []
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(ResBlock3D(ch, out_ch))
                ch = out_ch
            if level < len(channel_mult) - 1:
                blocks.append(Downsample3D(ch))
        self.blocks = nn.ModuleList(blocks)
        self.mid = nn.Sequential(ResBlock3D(ch, ch), ResBlock3D(ch, ch))
        self.out = nn.Sequential(norm(ch), nn.SiLU(), nn.Conv3d(ch, 2 * z_channels, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.mid(h)
        return self.out(h)
