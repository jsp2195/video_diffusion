from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def nested_get(d: dict, key: str, default=None):
    cur = d
    for p in key.split('.'):
        if p not in cur:
            return default
        cur = cur[p]
    return cur


def parse_args(default_config: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()
