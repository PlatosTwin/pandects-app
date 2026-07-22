#!/bin/bash
#SBATCH --job-name=mna-tax-deepseek
#SBATCH --account=torch_pr_744_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nmb9729/mna/taxonomy/deepseek/logs/%x-%j.out

# Offline batch taxonomy inference: runs batch_inference.py inside the
# prebuilt sglang apptainer image. No HTTP server — sglang.Engine loads the
# model in-process and batches all prompts internally.
#
# Override defaults at submit time, e.g.:
# sbatch --export=ALL,INPUT=data/smoke.jsonl,OUTPUT=data/smoke.out.jsonl run_inference.sbatch

# Old approach (separate taxonomy.json stitched at runtime):
# sbatch --export=ALL,INPUT=data/trial_100_2026-04-25.jsonl,OUTPUT=data/trial_100_2026-04-25.out.jsonl,TAXONOMY=data/taxonomy.json run_inference.sbatch

# New approach (ready-to-ingest JSONL with taxonomy baked per row — no TAXONOMY export):
# sbatch --export=ALL,INPUT=data/trial_100_2026_05_11.jsonl,OUTPUT=data/trial_100_2026_05_11.out.jsonl run_inference.sbatch

set -euo pipefail

DEEPSEEK_DIR="/scratch/nmb9729/mna/taxonomy/deepseek"
SIF_PATH="/scratch/nmb9729/overlays/sglang-runtime.sif"
LOG_DIR="${DEEPSEEK_DIR}/logs"

mkdir -p "${LOG_DIR}"

# Inputs (override via --export). Paths are relative to ${DEEPSEEK_DIR} unless absolute.
# The input JSONL is expected to be ready-to-ingest (per-row `messages` already
# include the taxonomy in the system prompt) — no separate taxonomy file.
INPUT="${INPUT:-data/batch.jsonl}"
OUTPUT="${OUTPUT:-data/batch.out.jsonl}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Pro}"

# Engine tuning — mirror the sglang serve invocation we use elsewhere.
TP_SIZE="${TP_SIZE:-8}"
MEM_FRACTION="${MEM_FRACTION:-0.88}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-marlin}"

# MAX_NEW_TOKENS unset => batch_inference.py auto-scales it from sections/row.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

bool_flag() {
  # bool_flag <flag-name> <value>  -> "--flag" or "--no-flag"
  if [[ "$2" == "1" || "$2" == "true" ]]; then echo "--$1"; else echo "--no-$1"; fi
}

EXTRA_ARGS=()
EXTRA_ARGS+=("$(bool_flag trust-remote-code "${TRUST_REMOTE_CODE}")")
EXTRA_ARGS+=(--moe-runner-backend "${MOE_RUNNER_BACKEND}")
[[ -n "${MAX_NEW_TOKENS}" ]] && EXTRA_ARGS+=(--max-new-tokens "${MAX_NEW_TOKENS}")

resolve() {
  case "$1" in
    /*) echo "$1" ;;
    *)  echo "${DEEPSEEK_DIR}/$1" ;;
  esac
}
INPUT_ABS="$(resolve "${INPUT}")"
OUTPUT_ABS="$(resolve "${OUTPUT}")"

echo "==== run_inference.sbatch ===="
echo "job_id=${SLURM_JOB_ID:-local} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?}"
echo "model=${MODEL} tp=${TP_SIZE} mem_fraction=${MEM_FRACTION} max_running_requests=${MAX_RUNNING_REQUESTS}"
echo "input=${INPUT_ABS}"
echo "output=${OUTPUT_ABS}"
echo "max_new_tokens=${MAX_NEW_TOKENS:-auto}"
echo "trust_remote_code=${TRUST_REMOTE_CODE} moe_runner=${MOE_RUNNER_BACKEND}"

if [[ ! -f "${INPUT_ABS}" ]]; then
  echo "ERROR: input file not found: ${INPUT_ABS}" >&2
  exit 2
fi

echo "[$(date -Is)] running batch_inference.py inside apptainer"
apptainer exec --nv \
  --bind "${DEEPSEEK_DIR}:${DEEPSEEK_DIR}" \
  --env SGLANG_APPLY_CONFIG_BACKUP=large \
  "${SIF_PATH}" \
  python3 "${DEEPSEEK_DIR}/batch_inference.py" \
    --input "${INPUT_ABS}" \
    --output "${OUTPUT_ABS}" \
    --model "${MODEL}" \
    --tp-size "${TP_SIZE}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --log-level "${LOG_LEVEL}" \
    "${EXTRA_ARGS[@]}"

echo "[$(date -Is)] done"
