# ASA Advertising-Compliance Legal LLM Pipeline

A retrieval-augmented, LoRA fine-tuned legal LLM for UK advertising-compliance
questions (CAP/BCAP Code + supporting legislation), built and trained on the
Nottingham CS GPU Cluster. Supplementary material for the MSc dissertation
*Mitigating Hallucination in Legal Large Language Models via RAG and
Fine-Tuning: A Dual-Perspective User Evaluation*.

Ask it about an advertisement; it answers with a verdict, the CAP/BCAP rules it
believes were breached, and the retrieved rule and case text those came from:

```
Decision: Upheld. Rules cited: CAP Code (Edition 12) rule 13.10.
...
```

## Contents

| Path | What it is |
| --- | --- |
| `code/` | Every script, described in [What each script does](#what-each-script-does) |
| `code/result/` | Archived `slurm-<job-id>.out` for every run reported in the dissertation |
| `data/raw/` | Scraped/downloaded source material. Legislation XML is `.gitignore`'d (large, and re-fetchable from legislation.gov.uk) |
| `data/processed/` | Per-section legislation JSONL, one file per Act/Regulation |
| `data/rag/` | `chunks.jsonl` + `chunk_embeddings.npy` (the RAG index) |
| `data/finetune/` | `train/valid/test.jsonl`, chat-format fine-tuning data |
| `adapters/` | Trained LoRA adapters (see [Adapters](#adapters)) |

## Adapters

Two adapters were trained:

- **`legal_lora_v2_gpu_bf16`** — trained on a class-imbalance-corrected split
  (minority "Not upheld" rows oversampled to 30% of the training set;
  validation and test left at their natural distribution). **Use this one.**
- `legal_lora_v1_gpu_bf16` — the original run, kept for comparison. It predates
  the correction and answers "Upheld" for almost any input.

Both are stored here at bfloat16 (67MB each). The float32 originals saved by
training are 133MB each, over GitHub's 100MB per-file limit; the per-checkpoint
optimizer state comes to about 8GB and is not distributed at all. Training ran
with bf16 compute and inference loads the base model in 4-bit NF4 with bf16
compute, so the float32 mantissa was never used at generation time (measured
change per weight: at most 6.1e-05 absolute, 0.39% relative). Outputs should be
equivalent, though not guaranteed bit-identical.

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

Everything under [Just run the model](#just-run-the-model) needs a GPU. The
data-collection and index-building scripts run fine on a laptop.

## What each script does

Run any script with `--help` for its full argument list. Scripts find their own
data directories relative to the repository, so they work from any working
directory; the `--adapter-path` you pass in is the one exception, being an
ordinary relative path. All examples below assume you are at the repository
root.

### Data collection — `data/raw/`

| Script | Purpose |
| --- | --- |
| `scrape_cap_bcap_code.py` | Scrapes the full CAP and BCAP Code rule text from asa.org.uk. No arguments. Writes `cap_code_sections.json`, `bcap_code_sections.json` (22 + 33 sections, 980 rules). |
| `scrape_asa_rulings.py` | Scrapes real ASA rulings by topic. Takes group names: `group_b` (medical/cosmetic), `group_d` (age-restricted/vulnerable-audience), or both. Writes `asa_rulings_group_{b,d}.jsonl` (540 rulings). |
| `xml_parser.py` | Parses legislation.gov.uk CLML XML into per-section JSONL. Takes XML paths and `-o`. Expects the XML already downloaded from the `/data.xml` endpoint. |

```bash
python code/scrape_cap_bcap_code.py
python code/scrape_asa_rulings.py group_b group_d
python code/xml_parser.py data/raw/*.xml -o data/processed/sections.jsonl
```

### Index and dataset building

| Script | Purpose |
| --- | --- |
| `build_rag_chunks.py` | Flattens all three sources into one `chunks.jsonl` (3,135 chunks) with a uniform schema. One rule = one chunk; one legislation section = one chunk; each ruling splits into a summary chunk and an assessment chunk. No arguments. |
| `build_embeddings.py` | Embeds every chunk with `BAAI/bge-small-en-v1.5` into `chunk_embeddings.npy`, row-aligned with `chunks.jsonl`. Takes 15--20s on CPU. No arguments. Re-run whenever chunks change. |
| `build_finetune_dataset.py` | Turns raw rulings into chat-format train/valid/test splits (80/10/10, fixed seed). Drops incomplete and over-length examples, then oversamples minority-decision rows in the training split only. No arguments. |

```bash
python code/build_rag_chunks.py
python code/build_embeddings.py
python code/build_finetune_dataset.py
```

### Training and evaluation

| Script | Purpose |
| --- | --- |
| `train_hf.py` | QLoRA fine-tuning. Requires `--output-dir`; `--max-steps 20` gives a quick smoke test. Submit via `train_job.sbatch` for the real run. |
| `evaluate_finetune_hf.py` | Scores the **base** and **fine-tuned** models on the held-out test set. Requires `--adapter-path`. Also defines the shared scoring logic the other two evaluation scripts import. |
| `evaluate_hybrid_hf.py` | Scores the **Hybrid** configuration (fine-tuned + retrieval). Requires `--adapter-path`. |
| `evaluate_rag_only_hf.py` | Scores the **RAG-only** configuration (retrieval on the un-fine-tuned base model). No adapter argument, by definition. |

All three evaluate against the same `data/finetune/test.jsonl` with the same
metrics — rule citation precision/recall/F1, fabricated rule rate, decision
match rate (with per-class recall), and format adherence — and print results as
soon as each pass finishes, so a timeout doesn't lose everything. Between them
they cover the four configurations compared in the dissertation: Baseline,
RAG-only, Fine-tuned, and Hybrid.

### Interactive use

| Script | Purpose |
| --- | --- |
| `retrieve.py` | Retrieval only, no generation. Takes a query, `--k`, `--group`, `--source-type`. Useful for checking what the index returns before involving the model. |
| `generate_hf.py` | Full retrieve-then-generate for one query. Takes a query, `--adapter-path`, `--k-rules`, `--k-cases`, `--max-tokens`. |
| `app.py` | Gradio web demo wrapping the same pipeline, answer and retrieved sources side by side. Takes `--adapter-path` (defaults to v2) and `--port`. |

## Just run the model

GPU work needs `srun`/`sbatch`, not a plain `python` call in your SSH session
(that ties up the terminal and dies on disconnect). The flags below are for the
`basic` account; re-check with `sacctmgr show associations user=$USER` if using
a different one, and always keep `--exclude=havok` (that node's GPU is too old
for the installed torch/bitsandbytes build).

Retrieval only, no GPU needed:

```bash
python code/retrieve.py "does a weight-loss ad breach the code if it claims a specific kg loss?"
```

One question, end to end:

```bash
srun --account=basic -c4 --mem=32G -G1 -p cs --exclude=havok --time=00:30:00 \
    python code/generate_hf.py \
        "A weight-loss supplement ad claims users will lose 10lbs in one week." \
        --adapter-path adapters/legal_lora_v2_gpu_bf16
```

Pass a plain description of the ad or behaviour, not a full question — the
"Does this breach the CAP Code...?" framing is added automatically, and adding
it yourself measurably dilutes retrieval relevance.

## Reproduce the results

Run in order; each stage only needs re-running if its inputs changed. Stages 1
and 2 are CPU-only.

```bash
# 1. collect data          (see "Data collection" above)
# 2. build index + dataset (see "Index and dataset building" above)

# 3. train
sbatch code/train_job.sbatch

# 4. evaluate all four configurations
sbatch code/eval_job.sbatch             # base vs. fine-tuned
sbatch code/hybrid_eval_job.sbatch      # fine-tuned + RAG
sbatch code/rag_only_eval_job.sbatch    # base + RAG

squeue --me                             # check progress
```

Results land in `slurm-<job-id>.out` in the directory the job was submitted
from. The runs behind the dissertation's reported numbers are in `code/result/`
— job 45273 is the v2 training run, 45274 and 45275 its fine-tuned and Hybrid
evaluations, and 45703 the RAG-only evaluation.

Smoke-test training first if you have changed anything about it, rather than
finding out eight hours later:

```bash
srun --account=basic -c4 --mem=32G -G1 -p cs --exclude=havok --time=00:15:00 \
    python code/train_hf.py --output-dir ~/asa_legal_pipeline_gpu/adapters/test_run --max-steps 20
```

## Run it for testers

For an evaluation session, where someone needs the system available for a
stretch rather than for one query, submit the demo as a batch job:

```bash
sbatch code/app_job.sbatch
squeue --me                  # wait for state R, note the job id
head -20 slurm-<job-id>.out  # prints the compute node name and port
```

The compute nodes have no outbound internet, so Gradio's `share=True` link does
not work here. `app.py` binds locally instead, and you reach it by forwarding
the port from your own machine through a login node to whichever compute node
the job landed on:

```bash
ssh -L 7860:<compute-node-name>:7860 <username>@jarvis.cs.nott.ac.uk
```

Then open `http://localhost:7860`. The tunnel stays open as long as that SSH
session does.

Practical notes for running sessions:

- The job stops at its `--time` limit (8h, the longest this partition allows in
  one go). For a longer study, resubmit — and expect a different compute node,
  so the tunnel command changes.
- Model loading takes a couple of minutes. Wait for `Ready` in the log before
  sending anyone the link.
- The first query after loading is slower than the rest; send one yourself
  before a session starts.
- Two participants sharing one instance will queue behind each other, since
  generation is sequential on a single GPU. Submit a second job on another port
  if that matters.
- `scancel <job-id>` when the session is over, rather than leaving a GPU idle
  and allocated.
