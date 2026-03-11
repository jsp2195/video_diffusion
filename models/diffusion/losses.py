import torch
import torch.nn.functional as F


def v_target(clean: torch.Tensor, noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    c_skip = 1.0 / (sigma[:, None, None, None, None] ** 2 + 1.0)
    c_out = -sigma[:, None, None, None, None] / torch.sqrt(sigma[:, None, None, None, None] ** 2 + 1.0)
    return (clean - c_skip * noisy) / (c_out + 1e-8)


def v_prediction_loss(pred_v: torch.Tensor, target_v: torch.Tensor, sigma: torch.Tensor, min_snr_gamma: float = 5.0) -> torch.Tensor:
    mse = F.mse_loss(pred_v, target_v, reduction="none")
    snr = 1.0 / (sigma[:, None, None, None, None] ** 2)
    weight = torch.minimum(snr, torch.full_like(snr, min_snr_gamma)) / (snr + 1e-8)
    return (weight * mse).mean()
