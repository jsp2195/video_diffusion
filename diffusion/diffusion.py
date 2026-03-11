import math
from dataclasses import dataclass

import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-5, 0.999).float()


@dataclass
class DiffusionSchedule:
    timesteps: int
    schedule: str = "cosine"

    def __post_init__(self):
        if self.schedule == "cosine":
            betas = cosine_beta_schedule(self.timesteps)
        elif self.schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.timesteps)
        else:
            raise ValueError("Unknown beta schedule")

        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device: torch.device):
        for name in [
            "betas",
            "alphas",
            "alphas_cumprod",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
        ]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

    def forward_noise(self, x0: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x0)
        alpha = self.sqrt_alphas_cumprod[t][:, None, None, None, None]
        sigma = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None]
        x_t = alpha * x0 + sigma * noise
        return x_t, noise

    def get_v(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha = self.sqrt_alphas_cumprod[t][:, None, None, None, None]
        sigma = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None]
        return alpha * noise - sigma * x0

    @torch.no_grad()
    def ddim_step(self, x_t: torch.Tensor, t: torch.Tensor, pred_v: torch.Tensor) -> torch.Tensor:
        alpha = self.sqrt_alphas_cumprod[t][:, None, None, None, None]
        sigma = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None]

        x0 = alpha * x_t - sigma * pred_v

        t_prev = (t - 1).clamp(min=0)
        alpha_prev = self.sqrt_alphas_cumprod[t_prev][:, None, None, None, None]
        sigma_prev = self.sqrt_one_minus_alphas_cumprod[t_prev][:, None, None, None, None]

        x_prev = alpha_prev * x0 + sigma_prev * pred_v

        mask = (t == 0).view(-1, 1, 1, 1, 1)
        return torch.where(mask, x0, x_prev)
