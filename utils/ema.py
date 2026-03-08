import copy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, update_after_step: int = 1000):
        self.decay = decay
        self.update_after_step = update_after_step
        self.step = 0
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.step += 1
        if self.step <= self.update_after_step:
            self.ema_model.load_state_dict(model.state_dict())
            return

        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return {
            "model": self.ema_model.state_dict(),
            "step": self.step,
            "decay": self.decay,
            "update_after_step": self.update_after_step,
        }

    def load_state_dict(self, state_dict):
        if "model" in state_dict:
            self.ema_model.load_state_dict(state_dict["model"])
            self.step = state_dict.get("step", 0)
            self.decay = state_dict.get("decay", self.decay)
            self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
        else:
            self.ema_model.load_state_dict(state_dict)
