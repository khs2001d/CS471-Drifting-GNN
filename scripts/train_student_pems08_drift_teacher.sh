#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CACHE_DIR="${CACHE_DIR:-outputs/pems08_teacher_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pems08_student_drift_teacher}"
PYTHON="${PYTHON:-python}"

"${PYTHON}" train_drift_teacher_student.py \
  --cache_dir "${CACHE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-4 \
  --num_student_samples 8 \
  --eta 0.1 \
  "$@"
