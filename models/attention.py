import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T,H,W]
        b, c, t, h, w = x.shape
        residual = x
        seq = x.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)  # [B*T,HW,C]
        seq = self.norm(seq)
        out, _ = self.attn(seq, seq, seq, need_weights=False)
        out = out.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        return residual + out


class TemporalAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T,H,W]
        b, c, t, h, w = x.shape
        residual = x
        seq = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)  # [B*H*W,T,C]
        seq = self.norm(seq)
        out, _ = self.attn(seq, seq, seq, need_weights=False)
        out = out.reshape(b, h, w, t, c).permute(0, 4, 3, 1, 2)
        return residual + out
