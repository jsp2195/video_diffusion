import torch
import torch.nn as nn

from models.autoencoder.blocks import ResBlock3D, Upsample3D, norm


class GrayVideoDecoder(nn.Module):
    def __init__(self, out_channels: int = 1, base_channels: int = 128, channel_mult=(1, 2, 4, 4), num_res_blocks: int = 2, z_channels: int = 4):
        super().__init__()
        ch = base_channels * channel_mult[-1]
        self.conv_in = nn.Conv3d(z_channels, ch, kernel_size=3, padding=1)
        self.mid = nn.Sequential(ResBlock3D(ch, ch), ResBlock3D(ch, ch))

        blocks = []
        rev_mult = list(channel_mult[::-1])
        for level, mult in enumerate(rev_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(ResBlock3D(ch, out_ch))
                ch = out_ch
            if level < len(rev_mult) - 1:
                blocks.append(Upsample3D(ch))
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Sequential(norm(ch), nn.SiLU(), nn.Conv3d(ch, out_channels, kernel_size=3, padding=1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid(h)
        for block in self.blocks:
            h = block(h)
        return self.out(h)
