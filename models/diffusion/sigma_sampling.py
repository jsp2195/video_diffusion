import torch


def sample_sigmas(batch_size: int, device: torch.device, sigma_min: float = 0.02, sigma_max: float = 1.0, rho: float = 7.0) -> torch.Tensor:
    u = torch.rand(batch_size, device=device)
    min_inv = sigma_min ** (1 / rho)
    max_inv = sigma_max ** (1 / rho)
    return (max_inv + u * (min_inv - max_inv)) ** rho


def karras_sigmas(num_steps: int, sigma_min: float = 0.02, sigma_max: float = 1.0, rho: float = 7.0, device: str = "cpu") -> torch.Tensor:
    ramp = torch.linspace(0, 1, num_steps, device=device)
    min_inv = sigma_min ** (1 / rho)
    max_inv = sigma_max ** (1 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device)], dim=0)
