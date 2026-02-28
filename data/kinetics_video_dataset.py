import glob
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def discover_and_split_videos(
    data_root: str,
    val_ratio: float,
    seed: int,
    max_videos: int = None,
) -> Tuple[List[str], List[str]]:
    pattern = os.path.join(data_root, "videos_val", "**", "*.mp4")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise RuntimeError(f"No mp4 files discovered at {pattern}")

    rng = random.Random(seed)
    rng.shuffle(files)

    if max_videos is not None and max_videos > 0:
        files = files[:max_videos]

    val_count = max(1, int(len(files) * val_ratio))
    val_files = files[:val_count]
    train_files = files[val_count:]
    if not train_files:
        train_files = files
        val_files = files[:1]

    print(f"Total mp4 files discovered: {len(files)}")
    print(f"Number assigned to train: {len(train_files)}")
    print(f"Number assigned to val: {len(val_files)}")
    print("First 3 example file paths:")
    for p in files[:3]:
        print(f"  {p}")

    return train_files, val_files


class KineticsVideoDataset(Dataset):
    def __init__(self, files: List[str], num_frames: int = 16, size: int = 128, max_seconds: int = 10):
        self.files = files
        self.num_frames = num_frames
        self.size = size
        self.max_seconds = max_seconds

    def __len__(self):
        return len(self.files)

    def _sample_indices(self, n: int, fps: float) -> np.ndarray:
        max_len = min(n, int(self.max_seconds * fps) if fps > 0 else n)
        max_len = max(max_len, self.num_frames)
        start_max = max(0, max_len - self.num_frames)
        start = np.random.randint(0, start_max + 1) if start_max > 0 else 0
        idx = np.arange(start, start + self.num_frames)
        idx = np.clip(idx, 0, n - 1)
        return idx

    def _resize_crop(self, frame: np.ndarray) -> np.ndarray:
        img = Image.fromarray(frame.astype(np.uint8))
        w, h = img.size
        scale = self.size / min(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        left = (new_w - self.size) // 2
        top = (new_h - self.size) // 2
        img = img.crop((left, top, left + self.size, top + self.size))
        return np.array(img)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        vr = VideoReader(path, ctx=cpu(0))
        n = len(vr)
        fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0

        if n <= 0:
            raise RuntimeError(f"Empty video: {path}")

        frame_idx = self._sample_indices(n, fps)
        frames = vr.get_batch(frame_idx).asnumpy()  # [T,H,W,3]

        proc = []
        for i in range(frames.shape[0]):
            f = self._resize_crop(frames[i])
            gray = _to_gray(f)
            gray = (gray / 127.5) - 1.0
            proc.append(gray.astype(np.float32))

        clip = np.stack(proc, axis=0)  # [T,H,W]
        clip = torch.from_numpy(clip).unsqueeze(0)  # [1,T,H,W]
        cond = clip[:, 0]  # [1,H,W]

        return {"cond": cond, "clip": clip, "path": path}
