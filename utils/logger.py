from typing import Optional


class TrainLogger:
    def __init__(self, enabled: bool = False, log_dir: str = "runs"):
        self.enabled = enabled
        self.writer = None
        if enabled:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, tag: str, value: float, step: int):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def add_video(self, tag: str, video, step: int, fps: int = 8):
        if self.writer is not None:
            self.writer.add_video(tag, video, step, fps=fps)

    def close(self):
        if self.writer is not None:
            self.writer.close()
