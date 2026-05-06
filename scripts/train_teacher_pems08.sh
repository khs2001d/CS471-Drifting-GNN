#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

python train.py \
  --data PEMS08 \
  --epsilon_theta UGnet \
  --N 100 \
  --sample_steps 100 \
  --ss ddpm \
  --T_h 12 \
  --T_p 12 \
  --n_samples 8 \
  --batch_size 8 \
  --is_test False \
  --teacher_checkpoint_path outputs/pems08_teacher/checkpoints/best.pt \
  "$@"
