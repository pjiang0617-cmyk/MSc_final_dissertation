#!/bin/bash
#SBATCH -J "legal-llm-hybrid-eval"
#SBATCH --account=basic
#SBATCH -c4
#SBATCH --mem=32G
#SBATCH -G1
#SBATCH -p cs
#SBATCH --exclude=havok
#SBATCH --time=01:30:00

# Slurm batch script for evaluate_hybrid_hf.py -- submit with:
#   sbatch hybrid_eval_job.sbatch
#
# Monitor:  squeue --me
# Cancel:   scancel --me
# Output:   slurm-<job-id>.out in the directory you submitted from

source ~/miniforge3/etc/profile.d/conda.sh
conda activate legal_llm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ~/asa_legal_pipeline_gpu/code/evaluate_hybrid_hf.py \
    --adapter-path ~/asa_legal_pipeline_gpu/adapters/legal_lora_v2_gpu
