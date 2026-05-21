#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CACHE_DIR="${CACHE_DIR:-outputs/pems08_teacher_cache_s32}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/pems08_student_drift_teacher_s32_antisymmetric/checkpoints/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pems08_student_drift_teacher_s32_antisymmetric_diversity_finetune}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-5}"
LAMBDA_DIVERSITY="${LAMBDA_DIVERSITY:-0.1}"

/home/ubuntu/miniconda3/envs/3d-seg/bin/python -u train_drift_teacher_student.py \
  --cache_dir "${CACHE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --init_checkpoint "${INIT_CHECKPOINT}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --num_student_samples 8 \
  --eta 0.1 \
  --loss_type antisymmetric_diversity \
  --lambda_diversity "${LAMBDA_DIVERSITY}" \
  "$@"
