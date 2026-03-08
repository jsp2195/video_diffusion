from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _matrix_sqrt_psd(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


class VideoFeatureExtractor:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = self._build_model().to(device).eval()

    def _build_model(self):
        model = None
        try:
            model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=True)
            model.blocks[-1] = torch.nn.Identity()
        except Exception:
            from torchvision.models.video import r3d_18, R3D_18_Weights

            model = r3d_18(weights=R3D_18_Weights.DEFAULT)
            model.fc = torch.nn.Identity()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    @torch.no_grad()
    def extract(self, videos: torch.Tensor) -> np.ndarray:
        # videos: [N,1,T,H,W] in [-1,1]
        x = videos.to(self.device)
        x = x.repeat(1, 3, 1, 1, 1)
        x = (x + 1.0) / 2.0
        x = F.interpolate(x, size=(x.shape[2], 224, 224), mode="trilinear", align_corners=False)
        feat = self.model(x)
        if feat.ndim > 2:
            feat = feat.flatten(1)
        return feat.detach().cpu().numpy()


def frechet_distance(real_feat: np.ndarray, gen_feat: np.ndarray) -> float:
    mu_r = real_feat.mean(axis=0)
    mu_g = gen_feat.mean(axis=0)
    cov_r = np.cov(real_feat, rowvar=False)
    cov_g = np.cov(gen_feat, rowvar=False)
    cov_mean = _matrix_sqrt_psd(cov_r @ cov_g)
    diff = mu_r - mu_g
    fvd = diff @ diff + np.trace(cov_r + cov_g - 2.0 * cov_mean)
    return float(np.real(fvd))


def compute_fvd(real_videos: torch.Tensor, gen_videos: torch.Tensor, device: torch.device) -> float:
    extractor = VideoFeatureExtractor(device=device)
    real_feat = extractor.extract(real_videos)
    gen_feat = extractor.extract(gen_videos)
    return frechet_distance(real_feat, gen_feat)
