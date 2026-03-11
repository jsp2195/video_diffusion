import torch


@torch.no_grad()
def euler_sample(denoiser, cond: dict, shape, sigmas: torch.Tensor, guidance_scale: float = 1.0, uncond: dict = None):
    x = torch.randn(shape, device=sigmas.device) * sigmas[0]
    for i in range(len(sigmas) - 1):
        sigma = torch.full((shape[0],), sigmas[i], device=sigmas.device)
        x0_c, _ = denoiser.denoise(x, sigma, cond)
        if uncond is not None and guidance_scale != 1.0:
            x0_u, _ = denoiser.denoise(x, sigma, uncond)
            x0 = x0_u + guidance_scale * (x0_c - x0_u)
        else:
            x0 = x0_c
        d = (x - x0) / (sigmas[i] + 1e-8)
        dt = sigmas[i + 1] - sigmas[i]
        x = x + d * dt
    return x
