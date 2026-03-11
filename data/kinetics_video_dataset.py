import glob
import os
import random
from typing import Dict, List, Tuple

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
) -> Tuple[List[str], List[str], List[str], str]:
    candidate_patterns = [
        os.path.join(data_root, "**", "*.mp4"),
        os.path.join(data_root, "**", "*.MP4"),
    ]
    files_set = set()
    for pattern in candidate_patterns:
        files_set.update(glob.glob(pattern, recursive=True))
    files_all = sorted(files_set)
    if not files_all:
        raise RuntimeError(f"No mp4 files discovered recursively under data_root={data_root}")

    files = list(files_all)
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
    print(f"Train count: {len(train_files)}")
    print(f"Val count: {len(val_files)}")
    print("First 3 file paths:")
    for p in files[:3]:
        print(f"  {p}")

    return train_files, val_files, files, data_root


class KineticsVideoDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        num_frames: int = 16,
        frame_stride: int = 1,
        size: int = 128,
        max_seconds: int = 10,
        cache_videos: bool = False,
    ):
        self.files = files
        if num_frames < 2:
            raise ValueError("num_frames must be >= 2 for first-frame-conditioned future prediction.")
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.size = size
        self.max_seconds = max_seconds
        self.cache_videos = cache_videos
        self._vr_cache: Dict[str, VideoReader] = {}
        self._meta_cache: Dict[str, Tuple[int, float]] = {}

    def __len__(self):
        return len(self.files)

    def _get_reader(self, path: str) -> VideoReader:
        if self.cache_videos and path in self._vr_cache:
            return self._vr_cache[path]
        vr = VideoReader(path, ctx=cpu(0))
        if self.cache_videos:
            self._vr_cache[path] = vr
        return vr

    def _get_meta(self, path: str, vr: VideoReader) -> Tuple[int, float]:
        if path in self._meta_cache:
            return self._meta_cache[path]
        n = len(vr)
        fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
        self._meta_cache[path] = (n, fps)
        return n, fps

    def _sample_indices(self, n: int, fps: float) -> np.ndarray:
        max_len = min(n, int(self.max_seconds * fps) if fps > 0 else n)
        max_len = max(max_len, self.num_frames)

        stride = self.frame_stride
        span = (self.num_frames - 1) * stride + 1
        max_start = max(0, max_len - span)
        start = np.random.randint(0, max_start + 1) if max_start > 0 else 0

        idx = start + np.arange(self.num_frames) * stride
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

    def __getitem__(self, idx):
        attempts = 0
        while attempts < len(self.files):
            path = self.files[idx]

            try:
                vr = self._get_reader(path)
                n, fps = self._get_meta(path, vr)

                if n < self.num_frames:
                    idx = (idx + 1) % len(self.files)
                    continue

                indices = self._sample_indices(n, fps)
                frames = vr.get_batch(indices).asnumpy()  # [T,H,W,3]

                processed = []
                for f in frames:
                    f = self._resize_crop(f)
                    gray = _to_gray(f)
                    processed.append(gray)

                frames = np.stack(processed)  # [T,H,W]

                frames = torch.from_numpy(frames).unsqueeze(0).float() / 127.5 - 1
                # [1,T,H,W]

                cond = frames[:, 0]  # [1,H,W]
                clip = frames        # [1,T,H,W]

                return {
                    "cond": cond,
                    "clip": clip,
                }

            except Exception:
                attempts += 1
                idx = (idx + 1) % len(self.files)

        raise RuntimeError("No valid videos could be decoded from dataset")
