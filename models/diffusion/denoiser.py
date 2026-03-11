import torch
import torch.nn as nn


class LatentDenoiser(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def precondition_in(self, x: torch.Tensor, sigma: torch.Tensor):
        return x / torch.sqrt(sigma[:, None, None, None, None] ** 2 + 1.0)

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, cond: dict):
        x_in = self.precondition_in(x, sigma)
        model_in = torch.cat([x_in, cond["concat"]], dim=1)
        v = self.network(model_in, sigma, cond["context"], cond["vector"])
        c_skip = 1.0 / (sigma[:, None, None, None, None] ** 2 + 1.0)
        c_out = -sigma[:, None, None, None, None] / torch.sqrt(sigma[:, None, None, None, None] ** 2 + 1.0)
        x0 = c_skip * x + c_out * v
        return x0, v
