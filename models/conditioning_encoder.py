import torch
import torch.nn as nn


def _group_count(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ConditioningEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid1 = max(out_channels // 4, 32)
        mid2 = max(out_channels // 2, 64)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid1, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(mid1), mid1),
            nn.SiLU(),
            nn.Conv2d(mid1, mid2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(mid2), mid2),
            nn.SiLU(),
            nn.Conv2d(mid2, out_channels, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, frame0: torch.Tensor) -> torch.Tensor:
        return self.net(frame0)
