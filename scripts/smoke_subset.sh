#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-./data/kinetics}
OUT_DIR=${2:-./outputs/smoke}

python train_video_ddpm.py \
  --data_root "${DATA_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --T 16 \
  --size 128 \
  --batch_size 1 \
  --max_videos 200 \
  --max_steps 30 \
  --shape_check \
  --num_workers 2
