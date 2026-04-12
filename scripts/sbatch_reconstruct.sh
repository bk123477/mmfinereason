#!/bin/bash
# ==============================================================
#  sbatch_reconstruct.sh — Batch reasoning reconstruction via SLURM
#
#  Processes all parquet files in DATASET_DIR using SLURM array jobs.
#  One SLURM task = one parquet file (full file, no part splitting).
#
#  Submit:
#    sbatch scripts/sbatch_reconstruct.sh
#
#  Modify the "Configuration" section below before submitting.
# ==============================================================

# ── SLURM directives ──────────────────────────────────────────
#SBATCH --job-name=sft_reconstruct
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-69%10          # 70 parquet files; %10 = max 10 concurrent
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
# #SBATCH --partition=gpu        # uncomment / change as needed
# #SBATCH --gres=gpu:0           # no GPU needed (API-based)

# ==============================================================
# ── Configuration — MODIFY THESE PATHS BEFORE SUBMITTING ─────
# ==============================================================

# Directory containing *.parquet files
DATASET_DIR="/path/to/dataset/raw/MMFineReason-SFT-586K-Qwen3-VL-235B-Thinking/data"

# Directory containing reconstruct_prompt.md and reasoning_workflows.md
PROMPTS_DIR="/path/to/prompts"

# Path to the sft_reconstruct.py script
PYTHON_SCRIPT="/path/to/src/sft_reconstruct/sft_reconstruct.py"

# Root output directory (results saved as OUTPUT_DIR/<parquet_num>/*.json)
OUTPUT_DIR="/path/to/output/reconstructed"

# OpenRouter model ID
MODEL="deepseek/deepseek-v3.2"

# Number of concurrent API calls per task
WORKERS=20

# OpenRouter API key (leave empty to read from .env next to PYTHON_SCRIPT)
OPENROUTER_API_KEY=""

# ==============================================================

set -euo pipefail

# Create log dir if it doesn't exist
mkdir -p logs

# Collect parquet files sorted alphabetically
mapfile -t PARQUETS < <(ls "${DATASET_DIR}"/*.parquet 2>/dev/null | sort)
N_PARQUETS=${#PARQUETS[@]}

if [[ ${N_PARQUETS} -eq 0 ]]; then
    echo "[ERROR] No parquet files found in: ${DATASET_DIR}"
    exit 1
fi

if [[ ${SLURM_ARRAY_TASK_ID} -ge ${N_PARQUETS} ]]; then
    echo "[SKIP] Task ${SLURM_ARRAY_TASK_ID} >= total parquets (${N_PARQUETS}). Nothing to do."
    exit 0
fi

PARQUET="${PARQUETS[$SLURM_ARRAY_TASK_ID]}"
echo "============================================================"
echo "Task ID   : ${SLURM_ARRAY_TASK_ID}"
echo "Parquet   : ${PARQUET}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Model     : ${MODEL}"
echo "Workers   : ${WORKERS}"
echo "============================================================"

# Build API key argument (prefer env var, fall back to .env auto-load in script)
API_KEY_ARG=()
if [[ -n "${OPENROUTER_API_KEY}" ]]; then
    API_KEY_ARG=(--api-key "${OPENROUTER_API_KEY}")
fi

python "${PYTHON_SCRIPT}" \
    --model              "${MODEL}" \
    --parquet            "${PARQUET}" \
    --reconstruct-prompt "${PROMPTS_DIR}/reconstruct_prompt.md" \
    --workflows          "${PROMPTS_DIR}/reasoning_workflows.md" \
    --output-dir         "${OUTPUT_DIR}" \
    --workers            "${WORKERS}" \
    "${API_KEY_ARG[@]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: DONE"
