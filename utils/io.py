import os

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


def denorm_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().byte().numpy()
    return x


def save_cond_png(cond: torch.Tensor, path: str):
    arr = denorm_to_uint8(cond)[0, 0]  # [H,W]
    Image.fromarray(arr, mode="L").save(path)


def save_mp4(clip: torch.Tensor, path: str, fps: int = 8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = denorm_to_uint8(clip)[0, 0]  # [T,H,W]
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for t in range(arr.shape[0]):
        frame = arr[t]
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        frame = frame.astype(np.uint8)
        writer.append_data(frame)
    writer.close()
