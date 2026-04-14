#!/bin/bash
# ==============================================================
#  retry_2step_errors_part.sh — Retry failed samples sequentially
#                               across all parquets for one part
#                               in the VLM -> LLM 2-step pipeline
#
#  Usage:
#    bash scripts/retry_2step_errors_part.sh <part>
# ==============================================================

# ── Configuration — UPDATE THESE FOR YOUR NEW DATASET ─────────

DATASET_DIR="/Users/hongminki/Downloads/SKT/mmfinereason/dataset/raw/MMFineReason-SFT-123K/data"
PROMPTS_DIR="/Users/hongminki/Downloads/SKT/mmfinereason/prompts"
PYTHON_SCRIPT="/Users/hongminki/Downloads/SKT/mmfinereason/src/sft_run/sft_2step_pipeline.py"
OUTPUT_DIR="/Users/hongminki/Downloads/SKT/mmfinereason/dataset/two_step_pipeline_123k"

VLM_MODEL="qwen/qwen3.5-397b-a17b"
LLM_MODEL="deepseek/deepseek-v3.2"
WORKERS=8
NUM_PARTS=4

START_IDX=0
END_IDX=69

# ──────────────────────────────────────────────────────────────

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

PARQUETS=( $(ls "${DATASET_DIR}"/*.parquet 2>/dev/null | sort) )
N_TOTAL=${#PARQUETS[@]}

if [[ ${N_TOTAL} -eq 0 ]]; then
    echo "[ERROR] No parquet files found in: ${DATASET_DIR}"
    exit 1
fi

echo "============================================================"
echo "Mode      : retry-errors (2-step pipeline)"
echo "Part      : ${PART} / $(( NUM_PARTS - 1 ))"
echo "Range     : ${START_IDX} → ${END_IDX}  (total parquets found: ${N_TOTAL})"
echo "VLM Model : ${VLM_MODEL}"
echo "LLM Model : ${LLM_MODEL}"
echo "Workers   : ${WORKERS}"
echo "Output    : ${OUTPUT_DIR}"
echo "============================================================"

SKIPPED=0
RETRIED=0

for (( idx=START_IDX; idx<=END_IDX; idx++ )); do
    if [[ ${idx} -ge ${N_TOTAL} ]]; then
        echo "[WARN] idx=${idx} >= total parquets (${N_TOTAL}). Stopping."
        break
    fi

    PARQUET="${PARQUETS[$idx]}"
    PARQUET_NAME="$(basename "${PARQUET}")"
    STEM=$(echo "${PARQUET_NAME%.parquet}" | cut -d'-' -f1,2 | tr '-' '_')
    PADDED=$(printf "%05d" "${idx}")
    ERROR_LOG="${OUTPUT_DIR}/${PADDED}/logs/${STEM}_part${PART}_errors.json"

    if [[ ! -f "${ERROR_LOG}" ]]; then
        echo "[SKIP] ${PARQUET_NAME} part${PART} — no error log"
        (( SKIPPED++ )) || true
        continue
    fi

    ERROR_COUNT=$(python3 -c "
import json
try:
    d = json.load(open('${ERROR_LOG}'))
    print(len(d))
except:
    print(0)
")

    if [[ "${ERROR_COUNT}" -eq 0 ]]; then
        echo "[SKIP] ${PARQUET_NAME} part${PART} — 0 errors in log"
        (( SKIPPED++ )) || true
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    printf "[%s] parquet %05d  part %d  errors=%s  →  %s\n" \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${idx}" "${PART}" "${ERROR_COUNT}" "${PARQUET_NAME}"
    echo "------------------------------------------------------------"

    python "${PYTHON_SCRIPT}" \
        --vlm-model          "${VLM_MODEL}" \
        --llm-model          "${LLM_MODEL}" \
        --parquet            "${PARQUET}" \
        --template-path      "${PROMPTS_DIR}/reasoning_distillation.md" \
        --reconstruct-prompt "${PROMPTS_DIR}/reconstruct_prompt.md" \
        --workflows          "${PROMPTS_DIR}/reasoning_workflows.md" \
        --output-dir         "${OUTPUT_DIR}" \
        --workers            "${WORKERS}" \
        --num-parts          "${NUM_PARTS}" \
        --part               "${PART}" \
        --retry-errors

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] parquet ${idx} part ${PART} retry done."
    (( RETRIED++ )) || true
done

echo ""
echo "============================================================"
echo "Retry-errors part ${PART} finished."
echo "  Retried : ${RETRIED} parquets"
echo "  Skipped : ${SKIPPED} parquets (no errors)"
echo "============================================================"
