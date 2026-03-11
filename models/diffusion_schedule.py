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
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), self.alpha_bar[:-1]], dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def to(self, device: torch.device):
        for name in [
            "betas",
            "alphas",
            "alpha_bar",
            "alpha_bar_prev",
            "sqrt_alpha_bar",
            "sqrt_one_minus_alpha_bar",
        ]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

    def _expand(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        a = self.sqrt_alpha_bar[t].view(b, 1, 1, 1, 1)
        s = self.sqrt_one_minus_alpha_bar[t].view(b, 1, 1, 1, 1)
        return a, s

    def forward_noise(self, x0: torch.Tensor, t: torch.Tensor, noise_offset: float = 0.0):
        noise = torch.randn_like(x0)
        if noise_offset > 0:
            noise = noise + noise_offset * torch.randn(x0.shape[0], x0.shape[1], 1, 1, 1, device=x0.device)
        a, s = self._expand(x0, t)
        xt = a * x0 + s * noise
        return xt, noise

    def velocity_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a, s = self._expand(x0, t)
        return a * noise - s * x0

    def predict_x0_from_v(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a, s = self._expand(xt, t)
        return a * xt - s * v

    def predict_eps_from_v(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a, s = self._expand(xt, t)
        return s * xt + a * v

    @torch.no_grad()
    def ddim_step_from_v(self, x_t: torch.Tensor, pred_v: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, eta: float = 0.0, dynamic_threshold: bool = False) -> torch.Tensor:
        b = x_t.shape[0]
        alpha_bar_t = self.alpha_bar[t].view(b, 1, 1, 1, 1)
        alpha_bar_prev = self.alpha_bar[t_prev.clamp(min=0)].view(b, 1, 1, 1, 1)

        x0_pred = self.predict_x0_from_v(x_t, pred_v, t)
        if dynamic_threshold:
            s = torch.quantile(x0_pred.abs().reshape(x0_pred.shape[0], -1), 0.995, dim=1)
            s = torch.maximum(s, torch.ones_like(s))
            s = s[:, None, None, None, None]
            x0_pred = torch.clamp(x0_pred, -s, s) / s
        eps_pred = self.predict_eps_from_v(x_t, pred_v, t)

        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(
            (1 - alpha_bar_t / alpha_bar_prev).clamp(min=0.0)
        )
        noise = torch.randn_like(x_t)
        dir_xt = torch.sqrt((1 - alpha_bar_prev - sigma**2).clamp(min=0.0)) * eps_pred
        x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

        zero_mask = (t_prev < 0).float().view(b, 1, 1, 1, 1)
        return x_prev * (1 - zero_mask) + x0_pred * zero_mask
