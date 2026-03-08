import math

import torch
import torch.nn as nn


class FramePositionalEncoding(nn.Module):
    def __init__(self, channels: int, max_frames: int = 512):
        super().__init__()
        self.channels = channels
        self.max_frames = max_frames

    def _build(self, t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(t, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.channels, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.channels)
        )
        pe = torch.zeros(t, self.channels, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T,H,W]
        b, c, t, h, w = x.shape
        pe = self._build(t, x.device, x.dtype).transpose(0, 1).view(1, c, t, 1, 1)
        return x + pe
