#!/bin/bash
#SBATCH -J "legal-llm-eval"
#SBATCH --account=basic
#SBATCH -c4
#SBATCH --mem=32G
#SBATCH -G1
#SBATCH -p cs
#SBATCH --exclude=havok
#SBATCH --time=02:00:00

# Slurm batch script for evaluate_finetune_hf.py -- submit with:
#   sbatch eval_job.sbatch
#
# Monitor:  squeue --me
# Cancel:   scancel --me   (or scancel <job-id> from squeue)
# Output:   slurm-<job-id>.out in the directory you submitted from
#
# --time=02:00:00 is deliberately generous: an interactive run of just the base
# model's 53-example pass alone used nearly all of a 45-minute budget, and this
# script runs base then fine-tuned sequentially (never concurrently -- loading
# both models at once OOMs a 15.6GB card, see evaluate_finetune_hf.py).
# Baseline results are printed as soon as that pass finishes, before the
# fine-tuned pass starts, so a timeout here still leaves partial results in the
# .out file instead of losing everything.

source ~/miniforge3/etc/profile.d/conda.sh
conda activate legal_llm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ~/asa_legal_pipeline_gpu/code/evaluate_finetune_hf.py \
    --adapter-path ~/asa_legal_pipeline_gpu/adapters/legal_lora_v1_gpu
