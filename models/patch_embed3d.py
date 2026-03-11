import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: tuple[int, int, int] = (2, 4, 4)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x = self.proj(x)
        b, c, t, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        return tokens, (t, h, w)


class PatchDecode3D(nn.Module):
    def __init__(self, embed_dim: int, out_channels: int, patch_size: tuple[int, int, int] = (2, 4, 4)):
        super().__init__()
        self.proj = nn.ConvTranspose3d(embed_dim, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, tokens: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
        b, _, c = tokens.shape
        t, h, w = grid_size
        x = tokens.transpose(1, 2).reshape(b, c, t, h, w)
        return self.proj(x)
