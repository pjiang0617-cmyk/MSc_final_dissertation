# ASA Advertising-Compliance Legal LLM Pipeline

A retrieval-augmented, LoRA fine-tuned legal LLM for UK advertising-compliance
questions (CAP/BCAP Code + supporting legislation), built and trained on the
Nottingham CS GPU Cluster.

## Setup

SSH into a login node (only reachable from the university network):

```bash
ssh <username>@jarvis.cs.nott.ac.uk   # or homer/virgil/plato, equivalent
```

Copy this folder over if it isn't already on the cluster:

```bash
scp -r asa_legal_pipeline_gpu <username>@jarvis.cs.nott.ac.uk:~/
```

Create the environment. Use Miniforge/conda, not a plain `venv` -- a bare
`python -m venv` lacks the Python C headers `triton` needs for its runtime
kernel compilation, which the model's attention implementation relies on.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate legal_llm          # create it first if it doesn't exist yet:
                                   #   conda create -n legal_llm python=3.13

export PIP_REQUIRE_VIRTUALENV=false   # this cluster's pip refuses to install
                                       # into a conda env otherwise
pip install -r requirements.txt
```

## Directory layout

```
code/       all scripts
data/raw/           scraped/downloaded source material (CAP/BCAP HTML, ASA
                     rulings, legislation.gov.uk XML -- the XML is
                     .gitignore'd, re-fetch it rather than expecting it in git)
data/processed/      per-section legislation JSONL, one file per Act/Regulation
data/rag/            chunks.jsonl + chunk_embeddings.npy (the RAG index)
data/finetune/       train/valid/test.jsonl (chat-format fine-tuning data)
code/result/         archived slurm-<job-id>.out logs for every training and
                     evaluation run reported in the dissertation
adapters/            trained LoRA adapters (.gitignore'd: 133MB per adapter,
                     over GitHub's per-file limit, plus ~8GB of per-checkpoint
                     optimizer state; archived separately)
```

Two adapters were trained. `legal_lora_v1_gpu` is the original run;
`legal_lora_v2_gpu` is retrained on a class-imbalance-corrected training split
(minority "Not upheld" rows oversampled to 30% of the training set, validation
and test left untouched) and is the one the reported results use.

## Running the pipeline

Run these in order the first time; re-run only whichever stage's inputs
changed.

### 1. Data collection

```bash
python code/scrape_cap_bcap_code.py       # -> data/raw/{cap,bcap}_code_sections.json
python code/scrape_asa_rulings.py         # -> data/raw/asa_rulings_group_{b,d}.jsonl
python code/xml_parser.py                 # -> data/processed/*_sections.jsonl
```

`xml_parser.py` expects the legislation XML files already downloaded into
`data/raw/` from legislation.gov.uk's `/data.xml` endpoint.

### 2. Build the RAG index

```bash
python code/build_rag_chunks.py    # -> data/rag/chunks.jsonl
python code/build_embeddings.py    # -> data/rag/chunk_embeddings.npy
```

### 3. Build the fine-tuning dataset

```bash
python code/build_finetune_dataset.py   # -> data/finetune/{train,valid,test}.jsonl
```

### 4. Train

GPU jobs need `srun`/`sbatch`, not a plain `python` call in your SSH session
(that ties up your terminal and dies if you disconnect). Flags below are for
the `basic` account; re-check with `sacctmgr show associations user=$USER` if
using a different account, and always include `--exclude=havok` (that node's
GPU is too old for the installed torch/bitsandbytes build).

Smoke test first:

```bash
srun --account=basic -c4 --mem=32G -G1 -p cs --exclude=havok --time=00:15:00 \
    python code/train_hf.py --output-dir ~/asa_legal_pipeline_gpu/adapters/legal_lora_v1_gpu --max-steps 20
```

Then submit the real run as a background job:

```bash
sbatch code/train_job.sbatch
squeue --me                # check on it
```

### 5. Evaluate

All three scripts compare against the same held-out `data/finetune/test.jsonl`
using the same scoring logic, and print results as soon as each pass finishes
rather than only at the end (so a timeout doesn't lose everything):

```bash
sbatch code/eval_job.sbatch             # base vs. fine-tuned (no retrieval)
sbatch code/hybrid_eval_job.sbatch      # fine-tuned + RAG
sbatch code/rag_only_eval_job.sbatch    # base + RAG (no fine-tuning)
```

Together these cover the four configurations compared in the dissertation:
Baseline, RAG-only, Fine-tuned, and Hybrid. Results land in
`slurm-<job-id>.out` in whichever directory the job was submitted from; the
runs reported in the dissertation are archived in `code/result/`.

### 6. Interactive use

Retrieval only, no generation:

```bash
python code/retrieve.py "does a weight-loss ad breach the code if it claims a specific kg loss?"
```

Full retrieve-then-generate for one query at a time:

```bash
python code/generate_hf.py "A weight-loss supplement ad claims users will lose 10lbs in one week." \
    --adapter-path adapters/legal_lora_v1_gpu
```

(Pass a plain description of the ad/behaviour, not a full question -- the
"Does this breach the CAP Code...?" framing is added automatically.)

### 7. Web demo

```bash
srun --account=basic -c4 --mem=32G -G1 -p cs --exclude=havok --time=02:00:00 \
    python -u code/app.py
```

`share=True` doesn't work here -- compute nodes have no outbound internet, so
`app.py` binds locally (`server_name="0.0.0.0"`) instead. To view it, forward
the port from your own machine through the login node to whichever compute
node the job landed on (check with `squeue --me`):

```bash
ssh -L 7860:<compute-node-name>:7860 <username>@jarvis.cs.nott.ac.uk
```

Then open `http://localhost:7860` in a browser.
