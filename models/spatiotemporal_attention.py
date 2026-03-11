import torch
import torch.nn as nn


class SpatiotemporalAttention(nn.Module):
    """Residual attention stack: spatial -> temporal -> cross(frame0)."""

    def __init__(self, channels: int, cond_channels: int, heads: int = 8):
        super().__init__()
        self.channels = channels

        self.spatial_norm = nn.LayerNorm(channels)
        self.spatial_attn = nn.MultiheadAttention(channels, heads, batch_first=True)

        self.temporal_norm = nn.LayerNorm(channels)
        self.temporal_attn = nn.MultiheadAttention(channels, heads, batch_first=True)

        self.cond_proj = nn.Linear(cond_channels, channels)
        self.cross_norm = nn.LayerNorm(channels)
        self.cross_attn = nn.MultiheadAttention(channels, heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels),
        )

    def _to_spatial_seq(self, x: torch.Tensor):
        b, c, t, h, w = x.shape
        seq = x.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)
        return seq, b, c, t, h, w

    def _from_spatial_seq(self, seq: torch.Tensor, b: int, c: int, t: int, h: int, w: int):
        return seq.view(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()

    def _to_temporal_seq(self, x: torch.Tensor):
        b, c, t, h, w = x.shape
        seq = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)
        return seq, b, c, t, h, w

    def _from_temporal_seq(self, seq: torch.Tensor, b: int, c: int, t: int, h: int, w: int):
        return seq.view(b, h, w, t, c).permute(0, 4, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        # Spatial attention: [B,C,T,H,W] -> [B*T, H*W, C]
        sseq, b, c, t, h, w = self._to_spatial_seq(x)
        s = self.spatial_norm(sseq)
        s_out, _ = self.spatial_attn(s, s, s, need_weights=False)
        sseq = sseq + s_out
        x = self._from_spatial_seq(sseq, b, c, t, h, w)

        # Temporal attention: [B,C,T,H,W] -> [B*H*W, T, C]
        tseq, b, c, t, h, w = self._to_temporal_seq(x)
        q = self.temporal_norm(tseq)
        t_out, _ = self.temporal_attn(q, q, q, need_weights=False)
        tseq = tseq + t_out

        cond = self.cond_proj(cond_tokens)
        cond = cond.unsqueeze(1).expand(b, h * w, cond.shape[1], c).reshape(b * h * w, cond.shape[1], c)
        q_cross = self.cross_norm(tseq)
        c_out, _ = self.cross_attn(q_cross, cond, cond, need_weights=False)
        tseq = tseq + c_out
        tseq = tseq + self.ff(tseq)

        return self._from_temporal_seq(tseq, b, c, t, h, w)
