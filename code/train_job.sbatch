#!/bin/bash
#SBATCH -J "legal-llm-lora"
#SBATCH --account=basic
#SBATCH -c4
#SBATCH --mem=32G
#SBATCH -G1
#SBATCH -p cs
#SBATCH --exclude=havok
#SBATCH --time=08:00:00

# Slurm batch script for train_hf.py -- submit with:
#   sbatch train_job.sbatch                 # full run
#   sbatch train_job.sbatch --max-steps 20  # smoke test (args pass through via $@)
#
# Monitor:  squeue --me
# Cancel:   scancel --me   (or scancel <job-id> from squeue)
# Output:   slurm-<job-id>.out in the directory you submitted from
#
# --mem=32G is host RAM headroom; the real constraint is GPU VRAM (~15.6GB on
# these cards), tuned via batch-size/grad-accum/max-seq-length/
# PYTORCH_CUDA_ALLOC_CONF below. See README.md for cluster access details
# (account/partition, excluded nodes, conda vs venv).

source ~/miniforge3/etc/profile.d/conda.sh
conda activate legal_llm
export PIP_REQUIRE_VIRTUALENV=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ~/asa_legal_pipeline_gpu/code/train_hf.py \
    --output-dir ~/asa_legal_pipeline_gpu/adapters/legal_lora_v2_gpu \
    --batch-size 1 \
    --grad-accum 8 \
    --max-seq-length 1024 \
    "$@"
