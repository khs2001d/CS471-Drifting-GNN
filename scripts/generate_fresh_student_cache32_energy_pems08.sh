#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
CHECKPOINT="${CHECKPOINT:-outputs/pems08_student_energy_finetune_s32_full/checkpoints/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pems08_eval_fresh_refs/student_cache32_energy_samples32}"
PYTHON="${PYTHON:-python}"
"${PYTHON}" -u generate_fresh_pems08_samples.py   --model_type student   --model_name Student_cache32_energy   --checkpoint "${CHECKPOINT}"   --output_dir "${OUTPUT_DIR}"   --sample_steps 1   --num_samples 32   --batch_size "${BATCH_SIZE:-32}"   "$@"
