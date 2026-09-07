#!/bin/bash
#SBATCH -J "legal-llm-app"
#SBATCH --account=basic
#SBATCH -c4
#SBATCH --mem=32G
#SBATCH -G1
#SBATCH -p cs
#SBATCH --exclude=havok
#SBATCH --time=08:00:00

# Long-running Gradio demo, for evaluation sessions where a participant needs
# the system available for a while. Submit with:
#   sbatch app_job.sbatch
#
# Unlike srun, this survives the SSH session that submitted it disconnecting.
# It stops when the job hits --time (8h, the longest this partition allows in
# one go); resubmit for a longer study and expect a different compute node.
#
# Monitor:  squeue --me
# Cancel:   scancel <job-id>
# Output:   slurm-<job-id>.out in the directory you submitted from -- the node
#           name and port needed for the SSH tunnel are printed at the top.

source ~/miniforge3/etc/profile.d/conda.sh
conda activate legal_llm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PORT=7860

echo "============================================================"
echo "Serving on compute node: $(hostname)   port: ${PORT}"
echo "From your own machine, forward the port through a login node:"
echo "  ssh -L ${PORT}:$(hostname):${PORT} \$USER@jarvis.cs.nott.ac.uk"
echo "then open http://localhost:${PORT}"
echo "============================================================"

python -u ~/asa_legal_pipeline_gpu/code/app.py \
    --adapter-path ~/asa_legal_pipeline_gpu/adapters/legal_lora_v2_gpu_bf16 \
    --port ${PORT}
