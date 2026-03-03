#!/bin/bash
#SBATCH --job-name=ner-tune
#SBATCH --output=/scratch/nmb9729/mna/ner/logs/%j-ner-run.log
#SBATCH --error=/scratch/nmb9729/mna/ner/logs/%A_%a-ner-tune.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8gb
#SBATCH --gres=gpu:1
#SBATCH --time=28:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nmb9729@nyu.edu
#SBATCH --account=torch_pr_744_general

export PYTHONUNBUFFERED=1

singularity exec --nv \
	    --overlay /scratch/nmb9729/overlays/mna.ext3:ro \
	    /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif\
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/nmb9729/mna/ner && python ner.py tune --num-trials 12 --max-epochs 10"
