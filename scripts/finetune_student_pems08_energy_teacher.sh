#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CACHE_DIR="${CACHE_DIR:-outputs/pems08_teacher_cache}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/pems08_student_drift_teacher/checkpoints/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pems08_student_energy_finetune}"

python train_drift_teacher_student.py \
  --cache_dir "${CACHE_DIR}" \
  --init_checkpoint "${INIT_CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs 10 \
  --batch_size 16 \
  --lr 5e-5 \
  --num_student_samples 8 \
  --loss_type energy \
  "$@"
