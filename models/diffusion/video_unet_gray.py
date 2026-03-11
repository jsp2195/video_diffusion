import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_ch, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, channels: int, context_dim: int, heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.self_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(channels, heads, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(channels, channels * 4), nn.GELU(), nn.Linear(channels * 4, channels))

    def forward(self, x, context):
        b, c, t, h, w = x.shape
        y = self.norm(x).permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
        y = y + self.self_attn(y, y, y, need_weights=False)[0]
        y = y + self.cross_attn(y, context, context, need_weights=False)[0]
        y = y + self.ff(y)
        return y.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)


class VideoUNetGray(nn.Module):
    def __init__(self, in_channels=8, out_channels=4, model_channels=192, channel_mult=(1, 2, 4, 4), num_res_blocks=2, context_dim=512, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.model_channels = model_channels
        self.time_embed = nn.Sequential(nn.Linear(model_channels, model_channels * 4), nn.SiLU(), nn.Linear(model_channels * 4, model_channels * 4))
        self.vec_proj = nn.Linear(context_dim, model_channels * 4)
        self.input_conv = nn.Conv3d(in_channels, model_channels, 3, padding=1)

        self.down = nn.ModuleList()
        ch = model_channels
        self.skip_channels = []
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(ch, out_ch, model_channels * 4))
                ch = out_ch
                if mult >= 4:
                    self.down.append(AttnBlock(ch, context_dim))
                self.skip_channels.append(ch)
            if i != len(channel_mult) - 1:
                self.down.append(nn.Conv3d(ch, ch, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))

        self.mid = nn.ModuleList([ResBlock(ch, ch, model_channels * 4), AttnBlock(ch, context_dim), ResBlock(ch, ch, model_channels * 4)])

        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                skip_ch = self.skip_channels.pop()
                self.up.append(ResBlock(ch + skip_ch, out_ch, model_channels * 4))
                ch = out_ch
                if mult >= 4:
                    self.up.append(AttnBlock(ch, context_dim))
            if i != 0:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'), nn.Conv3d(ch, ch, 3, padding=1)))

        self.out = nn.Sequential(nn.GroupNorm(32, ch), nn.SiLU(), nn.Conv3d(ch, out_channels, 3, padding=1))

    def _run(self, module, *args):
        if self.use_checkpoint and self.training and any(a.requires_grad for a in args if torch.is_tensor(a)):
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, x, sigma, context, vector):
        t_emb = timestep_embedding(torch.log(sigma + 1e-6), self.model_channels)
        emb = self.time_embed(t_emb) + self.vec_proj(vector)

        h = self.input_conv(x)
        skips = []
        for m in self.down:
            if isinstance(m, ResBlock):
                h = self._run(m, h, emb)
                skips.append(h)
            elif isinstance(m, AttnBlock):
                h = self._run(m, h, context)
            else:
                h = m(h)

        for m in self.mid:
            if isinstance(m, ResBlock):
                h = self._run(m, h, emb)
            else:
                h = self._run(m, h, context)

        for m in self.up:
            if isinstance(m, ResBlock):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self._run(m, h, emb)
            elif isinstance(m, AttnBlock):
                h = self._run(m, h, context)
            else:
                h = m(h)
        return self.out(h)
