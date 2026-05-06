#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CHECKPOINT="${CHECKPOINT:-outputs/pems08_student_drift_teacher/checkpoints/best.pt}"

python eval_drift_teacher_student.py \
  --checkpoint "${CHECKPOINT}" \
  --num_samples 8 \
  "$@"
