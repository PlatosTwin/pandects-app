#!/bin/bash
#SBATCH --job-name=ner-grid
#SBATCH --output=/scratch/nmb9729/mna/logs/%A_%a-ner-grid.out
#SBATCH --error=/scratch/nmb9729/mna/logs/%A_%a-ner-grid.err
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-10

export PYTHONUNBUFFERED=1

singularity exec --nv \
	    --overlay /scratch/nmb9729/overlays/mna.ext3:ro \
	    /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif\
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/nmb9729/mna/code && python run_from_grid.py --row-id ${SLURM_ARRAY_TASK_ID} --git-commit testtest"
