import math
from dataclasses import dataclass

import torch


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.999).float()


@dataclass
class DiffusionSchedule:
    timesteps: int
    beta_schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def __post_init__(self):
        if self.beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.timesteps)
        elif self.beta_schedule == "linear":
            betas = linear_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
        else:
            raise ValueError(f"Unknown beta_schedule={self.beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), self.alpha_bar[:-1]], dim=0)

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas)

    def to(self, device: torch.device):
        for name in [
            "betas",
            "alphas",
            "alpha_bar",
            "alpha_bar_prev",
            "sqrt_alpha_bar",
            "sqrt_one_minus_alpha_bar",
            "sqrt_recip_alpha",
        ]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

    def forward_noise(self, x0: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x0)
        bsz = x0.shape[0]
        a = self.sqrt_alpha_bar[t].view(bsz, 1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bar[t].view(bsz, 1, 1, 1, 1)
        xt = a * x0 + b * noise
        return xt, noise

    @torch.no_grad()
    def ddpm_step(self, x_t: torch.Tensor, pred_noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bsz = x_t.shape[0]
        alpha_t = self.alphas[t].view(bsz, 1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(bsz, 1, 1, 1, 1)
        beta_t = self.betas[t].view(bsz, 1, 1, 1, 1)
        mean = (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise) / torch.sqrt(alpha_t)

        nonzero_mask = (t > 0).float().view(bsz, 1, 1, 1, 1)
        noise = torch.randn_like(x_t)
        return mean + nonzero_mask * torch.sqrt(beta_t) * noise

    @torch.no_grad()
    def ddim_step(self, x_t: torch.Tensor, pred_noise: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        bsz = x_t.shape[0]
        alpha_bar_t = self.alpha_bar[t].view(bsz, 1, 1, 1, 1)
        alpha_bar_prev = self.alpha_bar[t_prev.clamp(min=0)].view(bsz, 1, 1, 1, 1)

        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        noise = torch.randn_like(x_t)

        dir_xt = torch.sqrt((1 - alpha_bar_prev - sigma ** 2).clamp(min=0.0)) * pred_noise
        x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

        zero_mask = (t_prev < 0).float().view(bsz, 1, 1, 1, 1)
        return x_prev * (1 - zero_mask) + x0_pred * zero_mask
