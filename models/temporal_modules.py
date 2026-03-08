import torch
import torch.nn as nn


class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)
