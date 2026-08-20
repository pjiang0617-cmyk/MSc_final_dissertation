# GPU/CUDA counterpart to asa_legal_pipeline/

This folder exists because the school GPU cluster is NVIDIA/CUDA, and MLX
(used in `asa_legal_pipeline/`) is Apple Silicon-only -- it does not run here
at all. Everything CUDA-specific lives here instead of touching the Mac setup.

## What's shared vs. what's new

**Copied over unchanged** (framework-agnostic -- scraping, parsing, RAG chunking/embedding/retrieval, dataset building):
- `code/scrape_cap_bcap_code.py`, `xml_parser.py`, `scrape_asa_rulings.py`
- `code/build_rag_chunks.py`, `build_embeddings.py`, `retrieve.py`
- `code/build_finetune_dataset.py` -- one line changed: `MODEL_DIR` points at the
  HF Hub id `google/gemma-4-E4B-it` instead of the local MLX checkpoint path
  (same tokenizer/chat template either way -- verified: `google/gemma-4-E4B-it`
  exists on the Hub and produces token counts identical to the Mac run: 534 built,
  16 dropped >3072 tokens, 518 kept)
- `data/raw/`, `data/processed/`, `data/finetune/`, `data/rag/` -- pure data, copied as-is

**NOT copied** (Mac/MLX-specific, not usable on CUDA):
- `models/gemma-4-e4b-it-4bit/` -- MLX-quantized checkpoint. On the GPU side,
  `google/gemma-4-E4B-it` is downloaded fresh from HF Hub and quantized
  on-the-fly via bitsandbytes (QLoRA) instead.
- `adapters/legal_lora_v1/` -- the ~120-step MLX LoRA checkpoint from the Mac.
  Not directly loadable by HF PEFT (different serialization format). Training
  here starts fresh rather than resuming from it.

**Written new for this environment** (the actual "can't share" part):
- `code/train_hf.py` -- QLoRA fine-tuning via transformers+peft+bitsandbytes+trl,
  mirroring the same design decisions as the Mac's `mlx_lm.lora` run (same LoRA
  rank/target modules in spirit, same completion-only loss masking, same
  max-seq-length). **Not yet run against real hardware** -- written and
  sanity-checked (tokenizer/chat-template behavior verified on CPU) but the
  actual training loop needs a real GPU to validate. Run a `--max-steps 20`
  smoke test before committing to a full run.
- `code/generate_hf.py`, `code/evaluate_finetune_hf.py` -- CUDA equivalents of
  the Mac's `generate.py` / `evaluate_finetune.py`. Same retrieval and scoring
  logic, transformers-based generation instead of mlx_lm.

## Things caught the hard way, running on the real cluster

- **trl 1.9.2 removed `DataCollatorForCompletionOnlyLM`** (present in older trl
  versions, which is what `train_hf.py` was originally written against without
  being able to test it). The replacement is a plain `SFTConfig(assistant_only_loss=True)`
  flag -- confirmed by inspecting `SFTConfig.__init__`'s real signature on the
  cluster. This is actually more robust than the old approach: it masks based on
  the tokenizer's own chat-template role boundaries instead of needing an exact
  turn-marker string hardcoded (the old approach would have needed
  `<|turn>model\n` specifically -- this Gemma-4 checkpoint's chat template uses
  its own turn markers, NOT the older Gemma `<start_of_turn>role`/`<end_of_turn>`
  format, verified by inspecting `chat_template.jinja` from the Mac download).
- **The `basic` account cannot use the `general` partition** (default in the
  cluster docs' own examples) -- `srun`/`sbatch` need `--account=basic -p cs`
  instead. Found via `sacctmgr show associations user=$USER` + `sinfo`.
- **Node `havok` in the `cs` partition has a GTX 1080 Ti** (compute capability
  6.1) -- too old for the installed `torch==2.13.0+cu130` / bitsandbytes build,
  crashes with `Error named symbol not found ... ops.cu` while loading the
  model. `storm` (RTX 2080 Ti) and `dart` (RTX A4000) both work. Use
  `--exclude=havok` (already in `train_job.sbatch`).

## The actual cluster: Nottingham CS GPU Cluster (Slurm-based)

Per "GPU Cluster: Access and Documentation" (Michael Pound, 29/04/2026):

- **Login nodes** (any of these, pick a favourite): `jarvis.cs.nott.ac.uk`,
  `homer.cs.nott.ac.uk`, `virgil.cs.nott.ac.uk`, `plato.cs.nott.ac.uk`. SSH only,
  and only resolves from within the university network (on campus, or via the
  university's remote-access setup).
- **Storage**: code, this whole project folder, and adapter checkpoints go under
  `/home/<username>/` (not `/data/` -- that's for datasets, and our whole
  dataset is only ~30MB anyway, doesn't need special treatment). **Neither
  location is backed up** (redundant against drive failure, not against
  accidental deletion) -- copy trained adapters and results back to your own
  machine once done, don't treat cluster storage as the only copy.
- **Jobs run via Slurm**, not by just running `python train_hf.py` directly in
  your SSH session -- that ties up your terminal and dies if you disconnect.
  Use `sbatch` (see `code/train_job.sbatch`) for anything that should run
  unsupervised for hours.

### Setup

```bash
ssh <username>@jarvis.cs.nott.ac.uk        # or homer/virgil/plato, equivalent
mkdir -p ~/asa_legal_pipeline_gpu && cd ~/asa_legal_pipeline_gpu
# copy this whole folder over, e.g. from your own machine:
#   scp -r asa_legal_pipeline_gpu <username>@jarvis.cs.nott.ac.uk:~/

python3.13 -m venv venv                     # venv is the cluster's recommended approach, not conda --
source venv/bin/activate                    # our dependencies are all plain pip packages, no conda-specific need
pip install -r requirements.txt
```

### Running

```bash
# smoke test first -- confirms the environment + pipeline actually work before
# committing a real job to the queue (short enough to just srun interactively)
srun --account=basic -c4 --mem=16G -G1 -p cs --exclude=havok python code/train_hf.py \
    --output-dir ~/asa_legal_pipeline_gpu/adapters/legal_lora_v1_gpu --max-steps 20

# once that's confirmed working, submit the real run as a batch job
sbatch code/train_job.sbatch

# check on it
squeue --me
```

Note: the account/partition/exclude flags above are specific to psxpj11's
`basic` account on this cluster -- a different account tier might have access
to other partitions (e.g. `amp16`/`amp20`/`amp48`/`ada24`, which sound like
newer Ampere/Ada-generation GPU partitions by name, but weren't reachable from
the `basic` account when this was tested). Re-check with `sacctmgr show
associations user=$USER` if using a different account.
