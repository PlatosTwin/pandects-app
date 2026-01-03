#!/bin/bash
#SBATCH --job-name=ner
#SBATCH --output=/scratch/nmb9729/mna/logs/%j-ner-run.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8gb
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nmb9729@nyu.edu

export PYTHONUNBUFFERED=1

singularity exec --nv \
	    --overlay /scratch/nmb9729/overlays/mna.ext3:ro \
	    /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif\
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/nmb9729/mna/code && python ner.py"