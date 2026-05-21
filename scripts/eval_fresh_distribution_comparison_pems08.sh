#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
/home/ubuntu/miniconda3/envs/3d-seg/bin/python -u eval_fresh_distribution_comparison.py   --teacher100_dir "${TEACHER100_DIR:-outputs/pems08_eval_fresh_refs/teacher100_steps100_samples32}"   --teacher40_dir "${TEACHER40_DIR:-outputs/pems08_eval_fresh_refs/teacher40_steps40_samples32}"   --student_cache8_dir "${STUDENT_CACHE8_DIR:-outputs/pems08_eval_fresh_refs/student_cache8_samples32}"   --student_cache32_dir "${STUDENT_CACHE32_DIR:-outputs/pems08_eval_fresh_refs/student_cache32_samples32}"   --student_cache32_aux_dir "${STUDENT_CACHE32_AUX_DIR:-outputs/pems08_eval_fresh_refs/student_cache32_aux_samples32}"   --output_csv "${OUTPUT_CSV:-outputs/pems08_results/fresh_distribution_comparison_samples32.csv}"   --output_json "${OUTPUT_JSON:-outputs/pems08_results/fresh_distribution_comparison_samples32.json}"   "$@"
