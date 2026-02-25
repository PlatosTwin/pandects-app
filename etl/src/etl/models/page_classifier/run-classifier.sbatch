#!/bin/bash
#SBATCH --job-name=crf-classifier
#SBATCH --output=/scratch/nmb9729/mna/page_classifier/logs/%j-classifier-run.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8gb
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nmb9729@nyu.edu
#SBATCH --account=torch_pr_744_general

export PYTHONUNBUFFERED=1

singularity exec --nv \
	    --overlay /scratch/nmb9729/overlays/mna.ext3:ro \
	    /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif\
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/nmb9729/mna/page_classifier && python classifier.py"