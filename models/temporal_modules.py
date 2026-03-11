import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
        )

    def forward(self, x):
        b, c, t, h, w = x.shape

        x = x.permute(0, 3, 4, 2, 1)  # B,H,W,T,C
        x = x.reshape(b * h * w, t, c)

        x, _ = self.attn(x, x, x)

        x = x.reshape(b, h, w, t, c)
        x = x.permute(0, 4, 3, 1, 2)

        return x


class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.temporal_attn = TemporalAttention(channels, heads=4)
        self.conv = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.temporal_attn(x)
        return x + self.conv(x)
