import torch
import torch.nn as nn
import torch.nn.functional as F


class SDPA(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert dim % heads == 0
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            b, n, c = t.shape
            return t.view(b, n, self.heads, self.head_dim).transpose(1, 2)

        qh = reshape_heads(q)
        kh = reshape_heads(k)
        vh = reshape_heads(v)
        out = F.scaled_dot_product_attention(qh, kh, vh)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.dim)
        return self.to_out(out)


class FactorizedSpaceTimeBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_sp = nn.LayerNorm(dim)
        self.norm_tm = nn.LayerNorm(dim)
        self.norm_ca = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)

        self.spatial_attn = SDPA(dim, heads=heads)
        self.temporal_attn = SDPA(dim, heads=heads)
        self.cross_q = nn.Linear(dim, dim)
        self.cross_kv = nn.Linear(dim, dim * 2)
        self.cross_out = nn.Linear(dim, dim)
        self.heads = heads
        self.head_dim = dim // heads

        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 8))

    def _cross_attention(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        b, n, c = q.shape
        q = self.cross_q(q).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        kv = self.cross_kv(kv)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, n, c)
        return self.cross_out(out)

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor, t_emb: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        b, n, c = x.shape
        tp, hp, wp = grid
        mod = self.mod(t_emb)
        sh_sp, sc_sp, sh_tm, sc_tm, sh_ca, sc_ca, sh_ff, sc_ff = mod.chunk(8, dim=-1)

        x_sp = self.norm_sp(x) * (1 + sc_sp[:, None, :]) + sh_sp[:, None, :]
        x_sp = x_sp.view(b, tp, hp * wp, c).reshape(b * tp, hp * wp, c)
        x_sp = self.spatial_attn(x_sp)
        x = x + x_sp.view(b, tp, hp * wp, c).reshape(b, n, c)

        x_tm = self.norm_tm(x) * (1 + sc_tm[:, None, :]) + sh_tm[:, None, :]
        x_tm = x_tm.view(b, tp, hp, wp, c).permute(0, 2, 3, 1, 4).reshape(b * hp * wp, tp, c)
        x_tm = self.temporal_attn(x_tm)
        x_tm = x_tm.view(b, hp, wp, tp, c).permute(0, 3, 1, 2, 4).reshape(b, n, c)
        x = x + x_tm

        x_ca = self.norm_ca(x) * (1 + sc_ca[:, None, :]) + sh_ca[:, None, :]
        x = x + self._cross_attention(x_ca, cond_tokens)

        x_ff = self.norm_ff(x) * (1 + sc_ff[:, None, :]) + sh_ff[:, None, :]
        x = x + self.ff(x_ff)
        return x
