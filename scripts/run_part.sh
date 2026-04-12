#!/bin/bash
# ==============================================================
#  run_part.sh — Run one part sequentially across all parquets
#
#  Usage:
#    bash scripts/run_part.sh <part>
#
#  Example (open 4 terminals, one per part):
#    bash scripts/run_part.sh 0
#    bash scripts/run_part.sh 1
#    bash scripts/run_part.sh 2
#    bash scripts/run_part.sh 3
#
#  Each terminal handles one part and advances to the next parquet
#  automatically when the current one finishes.
# ==============================================================

# ── Configuration — MODIFY THESE BEFORE RUNNING ──────────────

DATASET_DIR="/path/to/dataset/raw/MMFineReason-SFT-586K-Qwen3-VL-235B-Thinking/data"
PROMPTS_DIR="/path/to/prompts"
PYTHON_SCRIPT="/path/to/src/sft_reconstruct/sft_reconstruct.py"
OUTPUT_DIR="/path/to/output/reconstructed"

MODEL="deepseek/deepseek-v3.2"
WORKERS=20
NUM_PARTS=4

# Parquet range (inclusive on both ends, zero-padded 5-digit indices)
START_IDX=2     # start from 00002 (0-based)
END_IDX=69      # last parquet index (00069)

# Set to "--save-image" to include image_b64 in output, or leave empty
SAVE_IMAGE_FLAG=""

# ==============================================================

set -euo pipefail

PART="${1:-}"
if [[ -z "${PART}" ]]; then
    echo "Usage: bash $0 <part>"
    echo "  part: 0, 1, 2, or 3"
    exit 1
fi

if ! [[ "${PART}" =~ ^[0-9]+$ ]] || [[ "${PART}" -ge "${NUM_PARTS}" ]]; then
    echo "[ERROR] part must be an integer in [0, ${NUM_PARTS})"
    exit 1
fi

# Collect parquet files sorted alphabetically
mapfile -t PARQUETS < <(ls "${DATASET_DIR}"/*.parquet 2>/dev/null | sort)
N_TOTAL=${#PARQUETS[@]}

if [[ ${N_TOTAL} -eq 0 ]]; then
    echo "[ERROR] No parquet files found in: ${DATASET_DIR}"
    exit 1
fi

echo "============================================================"
echo "Part      : ${PART} / $(( NUM_PARTS - 1 ))"
echo "Range     : ${START_IDX} → ${END_IDX}  (total parquets found: ${N_TOTAL})"
echo "Model     : ${MODEL}"
echo "Workers   : ${WORKERS}"
echo "Output    : ${OUTPUT_DIR}"
echo "Save image: ${SAVE_IMAGE_FLAG:-no}"
echo "============================================================"

for (( idx=START_IDX; idx<=END_IDX; idx++ )); do
    if [[ ${idx} -ge ${N_TOTAL} ]]; then
        echo "[WARN] idx=${idx} >= total parquets (${N_TOTAL}). Stopping."
        break
    fi

    PARQUET="${PARQUETS[$idx]}"
    PARQUET_NAME="$(basename "${PARQUET}")"

    echo ""
    echo "------------------------------------------------------------"
    printf "[%s] parquet %05d  part %d  →  %s\n" \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${idx}" "${PART}" "${PARQUET_NAME}"
    echo "------------------------------------------------------------"

    python "${PYTHON_SCRIPT}" \
        --model              "${MODEL}" \
        --parquet            "${PARQUET}" \
        --reconstruct-prompt "${PROMPTS_DIR}/reconstruct_prompt.md" \
        --workflows          "${PROMPTS_DIR}/reasoning_workflows.md" \
        --output-dir         "${OUTPUT_DIR}" \
        --workers            "${WORKERS}" \
        --num-parts          "${NUM_PARTS}" \
        --part               "${PART}" \
        ${SAVE_IMAGE_FLAG}

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] parquet ${idx} part ${PART} done."
done

echo ""
echo "============================================================"
echo "All done. Part ${PART} finished range [${START_IDX}, ${END_IDX}]."
echo "============================================================"
