import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _matrix_sqrt_psd(mat):
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


class SimpleVideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.AvgPool3d((1,2,2)),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.AvgPool3d((1,2,2)),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.GELU(),

            nn.AdaptiveAvgPool3d((1,1,1))
        )

    def forward(self, x):
        x = self.net(x)
        return x.flatten(1)


class VideoFeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.model = SimpleVideoEncoder().to(device).eval()

    @torch.no_grad()
    def extract(self, videos):

        x = videos.to(self.device)

        x = x.repeat(1,3,1,1,1)

        x = (x + 1.0) / 2.0

        x = F.interpolate(
            x,
            size=(x.shape[2],96,96),
            mode="trilinear",
            align_corners=False
        )

        feat = self.model(x)

        return feat.detach().cpu().numpy()


def frechet_distance(real_feat, gen_feat):

    mu_r = real_feat.mean(axis=0)
    mu_g = gen_feat.mean(axis=0)

    cov_r = np.cov(real_feat, rowvar=False)
    cov_g = np.cov(gen_feat, rowvar=False)

    cov_mean = _matrix_sqrt_psd(cov_r @ cov_g)

    diff = mu_r - mu_g

    fvd = diff @ diff + np.trace(cov_r + cov_g - 2.0 * cov_mean)

    return float(np.real(fvd))


def compute_fvd(real_videos, gen_videos, device):

    if real_videos.shape[0] < 2 or gen_videos.shape[0] < 2:
        return float("nan")

    extractor = VideoFeatureExtractor(device)

    real_feat = extractor.extract(real_videos)
    gen_feat = extractor.extract(gen_videos)

    return frechet_distance(real_feat, gen_feat)
