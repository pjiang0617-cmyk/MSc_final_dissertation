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
#   sbatch train_job.sbatch                          # full run
#   sbatch train_job.sbatch --max-steps 20            # smoke test (args pass through via $@)
#
# Monitor:  squeue --me
# Cancel:   scancel --me   (or scancel <job-id> from squeue)
# Output:   slurm-<job-id>.out in the directory you submitted from
#
# Notes from actually getting a run to complete on this cluster (psxpj11's "basic" account):
#   - the "general" partition rejects the "basic" account -- "cs" is the one this
#     account is actually authorised for (found via sacctmgr show associations + sinfo)
#   - node "havok" in the cs partition has a GTX 1080 Ti (compute capability 6.1),
#     too old for the installed torch build to run on -- "storm" (RTX 2080 Ti) and
#     "dart"/"tamar" (RTX A4000-class, ~15.6GB VRAM) work, hence the exclude
#   - the venv (bare python3.13) had no Python.h, which broke Triton's runtime JIT
#     compilation deep inside this model's attention code -- switched to a Miniforge/
#     conda env instead, whose Python bundles its own headers
#   - conda envs aren't a "real" virtualenv as far as pip's concerned on this cluster;
#     needed PIP_REQUIRE_VIRTUALENV=false to let pip install into it at all
#   - --mem=32G here is headroom on the CPU/host-RAM side; the real constraint is GPU
#     VRAM (~15.6GB on these cards), tuned via batch-size/grad-accum/max-seq-length/
#     PYTORCH_CUDA_ALLOC_CONF below, not by this Slurm flag
#   - a 20-step smoke test finally completed clean with: batch-size=1, grad-accum=8,
#     max-seq-length=1024, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. The
#     "Rules cited:" line was also moved to right after "Decision:" in
#     build_finetune_dataset.py (front of the assistant answer, not the end) so it
#     survives truncation even on examples over the 1024-token cap.
#   - this cluster's sbatch wraps vanilla Slurm with extra validation (see the
#     "injecting GPU monitoring script" message on submit) that appears to require
#     the #SBATCH block to start immediately after the shebang line, with nothing
#     -- not even comments -- before or inside it. Keep all explanatory comments
#     after the directive block, like here, or submission fails outright.

source ~/miniforge3/etc/profile.d/conda.sh
conda activate legal_llm
export PIP_REQUIRE_VIRTUALENV=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ~/asa_legal_pipeline_gpu/code/train_hf.py \
    --output-dir ~/asa_legal_pipeline_gpu/adapters/legal_lora_v1_gpu \
    --batch-size 1 \
    --grad-accum 8 \
    --max-seq-length 1024 \
    "$@"
