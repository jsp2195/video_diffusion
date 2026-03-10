import os

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


def denorm_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    return x.numpy()


def save_cond_png(cond: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    arr = denorm_to_uint8(cond)

    # Accept shapes like (1,1,H,W) or (1,H,W)
    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3:
        arr = arr[0]

    Image.fromarray(arr, mode="L").save(path)


def save_mp4(clip: torch.Tensor, path: str, fps: int = 8):
    """
    Accepts clip tensors in common diffusion layouts:
        (B,C,T,H,W)
        (C,T,H,W)
        (T,C,H,W)
        (T,H,W)
    Converts to iterable frames (H,W,3)
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    arr = denorm_to_uint8(clip)

    # Remove batch if present
    if arr.ndim == 5:
        arr = arr[0]

    # (C,T,H,W) -> (T,C,H,W)
    if arr.ndim == 4 and arr.shape[0] in [1, 3]:
        arr = np.transpose(arr, (1, 0, 2, 3))

    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)

    for frame in arr:

        # (C,H,W) -> (H,W,C)
        if frame.ndim == 3 and frame.shape[0] in [1, 3]:
            frame = np.transpose(frame, (1, 2, 0))

        # grayscale -> RGB
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)

        frame = frame.astype(np.uint8)

        writer.append_data(frame)

    writer.close()
