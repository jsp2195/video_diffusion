from __future__ import annotations

import argparse
from pathlib import Path


def _flatten(prefix: str, obj: dict, out: dict):
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten(key, v, out)
        else:
            out[key] = v


def load_yaml_config(path: str | None) -> dict:
    if not path:
        return {}
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML is required for --config support. Install with `pip install pyyaml`.") from e

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    flat = {}
    _flatten("", cfg, flat)

    mapping = {
        "training.batch_size": "batch_size",
        "training.lr": "lr",
        "training.max_steps": "max_steps",
        "training.epochs": "epochs",
        "training.timesteps": "timesteps",
        "training.num_workers": "num_workers",
        "model.frames": "T",
        "model.resolution": "size",
        "data.root": "data_root",
        "data.val_ratio": "val_ratio",
    }

    defaults = {}
    for k, v in flat.items():
        if k in mapping:
            defaults[mapping[k]] = v
    return defaults


def parse_with_config(parser: argparse.ArgumentParser):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, _ = pre.parse_known_args()
    cfg_defaults = load_yaml_config(known.config)
    parser.set_defaults(**cfg_defaults)
    parser.add_argument("--config", type=str, default=known.config)
    return parser.parse_args()
