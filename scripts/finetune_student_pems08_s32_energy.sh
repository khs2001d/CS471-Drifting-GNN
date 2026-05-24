#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CACHE_DIR="${CACHE_DIR:-outputs/pems08_teacher_cache_s32}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/pems08_student_drift_teacher_s32_antisymmetric/checkpoints/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pems08_student_energy_finetune_s32_full}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-5e-5}"
PYTHON="${PYTHON:-python}"

"${PYTHON}" -u train_drift_teacher_student.py \
  --cache_dir "${CACHE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --init_checkpoint "${INIT_CHECKPOINT}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --num_student_samples 32 \
  --loss_type energy \
  "$@"
