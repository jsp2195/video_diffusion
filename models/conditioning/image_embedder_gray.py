import torch
import torch.nn as nn


class GrayImageEmbedder(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 512, patch_size: int = 8, width: int = 128, depth: int = 4, heads: int = 8):
        super().__init__()
        self.patch = nn.Conv2d(in_channels, width, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(width, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, batch_first=True, dim_feedforward=embed_dim * 4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W]
        h = self.patch(x)
        h = h.flatten(2).transpose(1, 2)
        h = self.proj(h)
        return self.encoder(h)
