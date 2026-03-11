import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from xformers.ops import memory_efficient_attention

    XFORMERS_AVAILABLE = True
except Exception:
    XFORMERS_AVAILABLE = False


class TemporalAttention(nn.Module):
    def __init__(self, channels: int, cond_channels: int, heads: int = 8, use_xformers: bool = True):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.use_xformers = use_xformers and XFORMERS_AVAILABLE

        self.norm = nn.LayerNorm(channels)
        self.self_attn = nn.MultiheadAttention(channels, heads, batch_first=True)

        self.cond_proj = nn.Linear(cond_channels, channels)
        self.cross_norm = nn.LayerNorm(channels)
        self.cross_attn = nn.MultiheadAttention(channels, heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels),
        )

    def _reshape_in(self, x: torch.Tensor):
        b, c, t, h, w = x.shape
        seq = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)
        return seq, b, c, t, h, w

    def _reshape_out(self, seq: torch.Tensor, b: int, c: int, t: int, h: int, w: int):
        return seq.view(b, h, w, t, c).permute(0, 4, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        seq, b, c, t, h, w = self._reshape_in(x)

        q = self.norm(seq)
        self_out, _ = self.self_attn(q, q, q, need_weights=False)
        seq = seq + self_out

        cond = self.cond_proj(cond_tokens)
        cond = cond.unsqueeze(1).expand(b, h * w, cond.shape[1], c).reshape(b * h * w, cond.shape[1], c)

        q2 = self.cross_norm(seq)
        cross_out, _ = self.cross_attn(q2, cond, cond, need_weights=False)
        seq = seq + cross_out
        seq = seq + self.ff(seq)

        return self._reshape_out(seq, b, c, t, h, w)
