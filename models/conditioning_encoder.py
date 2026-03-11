from typing import Dict, Iterable, List

import torch
import torch.nn as nn


def _group_count(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class _ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditioningEncoder(nn.Module):
    """Lightweight multi-scale encoder for frame-0 conditioning.

    Returns a feature pyramid:
      - stem: full-resolution feature
      - down: list aligned to each U-Net down level resolution
      - mid: bottleneck-resolution feature
    """

    def __init__(self, in_channels: int, base_channels: int, channel_mult: Iterable[int]):
        super().__init__()
        mults = list(channel_mult)
        if not mults:
            raise ValueError("channel_mult must not be empty")

        self.level_channels: List[int] = [base_channels * m for m in mults]

        self.stem = _ConvGNAct(in_channels, base_channels, stride=1)

        self.level_projs = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        curr_ch = base_channels
        for i, out_ch in enumerate(self.level_channels):
            self.level_projs.append(nn.Conv2d(curr_ch, out_ch, kernel_size=1))
            if i < len(self.level_channels) - 1:
                next_ch = self.level_channels[i + 1]
                self.downsamplers.append(_ConvGNAct(curr_ch, next_ch, stride=2))
                curr_ch = next_ch

        self.mid_down = _ConvGNAct(curr_ch, curr_ch, stride=2)
        self.mid_proj = nn.Conv2d(curr_ch, self.level_channels[-1], kernel_size=1)

    def forward(self, frame0: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        x = self.stem(frame0)
        down_feats: List[torch.Tensor] = []

        for i, proj in enumerate(self.level_projs):
            down_feats.append(proj(x))
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)

        mid_feat = self.mid_proj(self.mid_down(x))
        return {
            "stem": down_feats[0],
            "down": down_feats,
            "mid": mid_feat,
        }
