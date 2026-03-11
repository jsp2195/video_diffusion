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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr = denorm_to_uint8(cond)

    if arr.ndim == 4:  # (B,C,H,W)
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"Expected cond tensor with shape (C,H,W) or (B,C,H,W), got {arr.shape}")

    if arr.shape[0] == 1:
        Image.fromarray(arr[0], mode="L").save(path)
    elif arr.shape[0] == 3:
        Image.fromarray(np.transpose(arr, (1, 2, 0)), mode="RGB").save(path)
    else:
        raise ValueError(f"Expected cond channels 1 or 3, got {arr.shape[0]}")


def _normalize_clip_layout(arr: np.ndarray) -> np.ndarray:
    """Return (T,H,W,C) uint8-friendly layout."""
    if arr.ndim == 5:  # (B,C,T,H,W)
        arr = arr[0]

    if arr.ndim == 4:
        # (C,T,H,W) -> (T,H,W,C)
        if arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 3, 0))
        # (T,C,H,W) -> (T,H,W,C)
        elif arr.shape[1] in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))
        # (T,H,W,1/3) already
        elif arr.shape[-1] in (1, 3):
            pass
        else:
            raise ValueError(f"Unrecognized 4D clip layout: {arr.shape}")
    elif arr.ndim == 3:  # (T,H,W)
        arr = arr[..., None]
    else:
        raise ValueError(f"Unsupported clip ndim={arr.ndim}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] != 3:
        raise ValueError(f"Expected channels=1 or 3, got {arr.shape[-1]}")

    return arr


def save_mp4(clip: torch.Tensor, path: str, fps: int = 8):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr = denorm_to_uint8(clip)
    arr = _normalize_clip_layout(arr)

    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    try:
        for frame in arr:
            writer.append_data(frame.astype(np.uint8))
    finally:
        writer.close()
