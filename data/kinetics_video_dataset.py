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
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def discover_and_split_videos(data_root: str, val_ratio: float, seed: int, max_videos: int = None) -> Tuple[List[str], List[str], List[str], str]:
    pattern = os.path.join(data_root, "videos_val", "**", "*.mp4")
    files_all = sorted(glob.glob(pattern, recursive=True))
    if not files_all:
        raise RuntimeError(f"No mp4 files discovered at {data_root}/videos_val")
    files = list(files_all)
    rng = random.Random(seed)
    rng.shuffle(files)
    if max_videos:
        files = files[:max_videos]
    val_count = max(1, int(len(files) * val_ratio))
    val_files, train_files = files[:val_count], files[val_count:]
    if not train_files:
        train_files, val_files = files, files[:1]
    return train_files, val_files, files, pattern


class KineticsVideoDataset(Dataset):
    def __init__(self, files: List[str], num_frames: int = 12, size: int = 96, max_seconds: int = 10, cache_videos: bool = False, motion_buckets: int = 32, cond_aug_mean: float = 0.02, cond_aug_std: float = 0.01):
        self.files = files
        self.num_frames = num_frames
        self.size = size
        self.max_seconds = max_seconds
        self.cache_videos = cache_videos
        self.motion_buckets = motion_buckets
        self.cond_aug_mean = cond_aug_mean
        self.cond_aug_std = cond_aug_std
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
        max_start = max(0, max_len - self.num_frames)
        start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        return np.clip(start + np.arange(self.num_frames), 0, n - 1)

    def _resize_crop(self, frame: np.ndarray) -> np.ndarray:
        img = Image.fromarray(frame.astype(np.uint8))
        w, h = img.size
        scale = self.size / min(w, h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img = img.resize((nw, nh), resample=Image.BILINEAR)
        left, top = (nw - self.size) // 2, (nh - self.size) // 2
        return np.array(img.crop((left, top, left + self.size, top + self.size)))

    def _motion_bucket(self, video: torch.Tensor) -> int:
        diffs = (video[:, 1:] - video[:, :-1]).abs().mean().item()
        normed = max(0.0, min(1.0, diffs / 0.4))
        return int(normed * (self.motion_buckets - 1))

    def __getitem__(self, idx):
        attempts = 0
        while attempts < len(self.files):
            path = self.files[idx]
            try:
                vr = self._get_reader(path)
                n, fps = self._get_meta(path, vr)
                if n < self.num_frames:
                    attempts += 1
                    idx = (idx + 1) % len(self.files)
                    continue
                frames = vr.get_batch(self._sample_indices(n, fps)).asnumpy()
                gray = []
                for f in frames:
                    gray.append(_to_gray(self._resize_crop(f))[..., None])
                frames = np.stack(gray)
                video = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 127.5 - 1.0

                cond_raw = video[:, 0]
                cond_aug = float(np.clip(np.random.normal(self.cond_aug_mean, self.cond_aug_std), 0.0, 0.2))
                cond_noisy = (cond_raw + torch.randn_like(cond_raw) * cond_aug).clamp(-1, 1)

                fps_id = int(max(0, min(31, round(fps / 2.0))))
                motion_bucket = self._motion_bucket(video)

                return {
                    "video": video,
                    "clip": video,
                    "cond": cond_raw,
                    "cond_frames": cond_noisy,
                    "cond_frames_without_noise": cond_raw,
                    "fps_id": torch.tensor(fps_id, dtype=torch.long),
                    "motion_bucket_id": torch.tensor(motion_bucket, dtype=torch.long),
                    "cond_aug": torch.tensor(cond_aug, dtype=torch.float32),
                    "path": path,
                }
            except Exception:
                attempts += 1
                idx = (idx + 1) % len(self.files)
        raise RuntimeError("No valid videos could be decoded from dataset")
