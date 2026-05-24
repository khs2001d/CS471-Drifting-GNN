#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CHECKPOINT="${CHECKPOINT:-outputs/pems08_teacher/checkpoints/best.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pems08_teacher_cache}"
NUM_TEACHER_SAMPLES="${NUM_TEACHER_SAMPLES:-8}"
SAMPLE_STEPS="${SAMPLE_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PYTHON="${PYTHON:-python}"

"${PYTHON}" generate_teacher_cache.py \
  --checkpoint "${CHECKPOINT}" \
  --output_root "${OUTPUT_ROOT}" \
  --splits train val \
  --num_teacher_samples "${NUM_TEACHER_SAMPLES}" \
  --sample_steps "${SAMPLE_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  "$@"
