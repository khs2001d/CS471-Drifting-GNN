#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CHECKPOINT="${CHECKPOINT:-outputs/pems08_teacher/checkpoints/best.pt}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
PYTHON="${PYTHON:-python}"

"${PYTHON}" eval_teacher_pems08.py \
  --checkpoint "${CHECKPOINT}" \
  --sample_steps 100 \
  --num_samples "${NUM_SAMPLES}" \
  "$@"

"${PYTHON}" eval_teacher_pems08.py \
  --checkpoint "${CHECKPOINT}" \
  --sample_steps 40 \
  --num_samples "${NUM_SAMPLES}" \
  "$@"
