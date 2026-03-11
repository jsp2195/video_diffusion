import torch
import torch.nn as nn

from models.autoencoder.decoder_gray_video import GrayVideoDecoder
from models.autoencoder.encoder_gray import GrayVideoEncoder


class AutoencoderKLGray(nn.Module):
    def __init__(self, in_channels=1, out_ch=1, z_channels=4, base_channels=128, channel_mult=(1, 2, 4, 4), num_res_blocks=2):
        super().__init__()
        self.encoder = GrayVideoEncoder(in_channels, base_channels, channel_mult, num_res_blocks, z_channels)
        self.decoder = GrayVideoDecoder(out_ch, base_channels, channel_mult, num_res_blocks, z_channels)
        self.z_channels = z_channels

    def encode(self, x: torch.Tensor):
        moments = self.encoder(x)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mean, logvar

    def sample(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mean, logvar = self.encode(x)
        z = self.sample(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.mean(torch.exp(logvar) + mean.pow(2) - 1.0 - logvar)
