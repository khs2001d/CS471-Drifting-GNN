#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
CHECKPOINT="${CHECKPOINT:-outputs/pems08_teacher/checkpoints/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pems08_eval_fresh_refs/teacher40_steps40_samples32}"
/home/ubuntu/miniconda3/envs/3d-seg/bin/python -u generate_fresh_pems08_samples.py   --model_type teacher   --model_name Teacher40   --checkpoint "${CHECKPOINT}"   --output_dir "${OUTPUT_DIR}"   --sample_steps 40   --num_samples 32   --batch_size "${BATCH_SIZE:-32}"   "$@"
