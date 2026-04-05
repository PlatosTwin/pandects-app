#!/bin/bash
#SBATCH --job-name=ner-grid
#SBATCH --output=/scratch/nmb9729/mna/ner/logs/%A_%a-ner-grid.out
#SBATCH --error=/scratch/nmb9729/mna/ner/logs/%A_%a-ner-grid.err
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=28:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nmb9729@nyu.edu
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_744_general

export PYTHONUNBUFFERED=1
export GRID_PATH=/scratch/nmb9729/mna/ner/configs/grid.csv
export FROZEN_CONFIG_PATH=/scratch/nmb9729/mna/ner/configs/frozen_experiment_config.yaml
export DATA_PATH="${DATA_PATH:-/scratch/nmb9729/mna/ner/data/ner-data-revamp.parquet}"
export SPLIT_VERSION="${SPLIT_VERSION:-ner_data_revamp_v1}"
export SEED="${SEED:-42}"
export GIT_COMMIT="${GIT_COMMIT:-unknown}"
export NUM_WORKERS="${NUM_WORKERS:-3}"

singularity exec --nv \
	    --overlay /scratch/nmb9729/overlays/mna.ext3:ro \
	    /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif\
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/nmb9729/mna/ner && python run_from_grid.py --row-id ${SLURM_ARRAY_TASK_ID} --split-version ${SPLIT_VERSION} --seed ${SEED} --git-commit ${GIT_COMMIT} --grid-path ${GRID_PATH} --frozen-config-path ${FROZEN_CONFIG_PATH} --data-path ${DATA_PATH} --num-workers ${NUM_WORKERS}"
