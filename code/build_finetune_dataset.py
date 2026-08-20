"""
GPU/CUDA variant -- otherwise identical to asa_legal_pipeline/code/build_finetune_dataset.py.
Only change: MODEL_DIR points at the HF Hub id (google/gemma-4-E4B-it) instead of the
local MLX-quantized checkpoint, since this folder has no models/ directory (that
checkpoint is Mac/MLX-specific and isn't used for GPU training -- see train_hf.py).

Phase 1 (part 2): turn the raw ASA ruling records (data/raw/asa_rulings_*.jsonl)
into instruction-tuning pairs in the {"messages": [...]} chat format both mlx_lm.lora
and HF trl's SFTTrainer expect for chat-style fine-tuning.

Design choice: the fine-tuned adapter is meant to teach *behaviour* (how to
reason from given ad copy to a rule-grounded verdict, and to always cite the
exact rule numbers), not to memorise facts -- the facts (rule text, statute
text) belong in the RAG index instead. That's why every assistant answer ends
with an explicit "Rules cited:" line: we want the model to learn the *habit*
of citing, which is what RQ2 (accuracy of citations) actually measures.

Usage:
    python build_finetune_dataset.py
Writes:
    ../data/finetune/train.jsonl
    ../data/finetune/valid.jsonl
    ../data/finetune/test.jsonl
"""

import json
import random
from pathlib import Path

from transformers import AutoTokenizer

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "finetune"
MODEL_DIR = "google/gemma-4-E4B-it"  # HF Hub id -- tokenizer/chat template only, same as the MLX checkpoint's

# Empirically, only ~3% of examples exceed this many tokens (checked against the
# real tokenizer), but those outliers are exactly what crashed mlx_lm.lora with
# OOM on a 16GB machine even with batch-size=1 and gradient checkpointing --
# mlx_lm truncates from the end at train time, which would cut off the "Rules
# cited:" line we specifically engineered to be at the end. Dropping outliers
# here (source) is safer than relying on runtime truncation (mlx_lm/tuner/trainer.py).
MAX_TOKENS = 3072

SYSTEM_PROMPT = (
    "You are a UK advertising-compliance assistant. You assess whether marketing "
    "communications comply with the CAP Code (non-broadcast) or BCAP Code (broadcast), "
    "the self-regulatory advertising codes enforced by the ASA. When you give a verdict, "
    "you must ground it in the facts given and cite the exact rule number(s) breached "
    "(e.g. 'CAP Code rule 3.1'). If the facts given are insufficient to reach a verdict, "
    "say so explicitly instead of guessing."
)

SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, valid, test
SEED = 42


def build_user_message(record):
    parts = []
    if record.get("ad_description"):
        parts.append(f"Advertisement:\n{record['ad_description']}")
    if record.get("issue"):
        parts.append(f"Complaint / issue raised:\n{record['issue']}")
    if not parts:
        return None
    parts.append(
        "Does this advertisement breach the CAP Code or BCAP Code? "
        "If so, which rule(s) and why?"
    )
    return "\n\n".join(parts)


def build_assistant_message(record):
    assessment = record.get("assessment")
    if not assessment:
        return None
    decision = record.get("decision") or "Upheld"
    rules = record.get("cited_rules") or []
    edition = record.get("code_edition") or "CAP Code"

    # Rules cited now comes right after Decision, BEFORE the detailed reasoning --
    # not at the end. Tokenizer truncation (both train_hf.py's apply_chat_template
    # and mlx_lm's runtime truncation) cuts from the end of the sequence, and on the
    # GPU we're already forced down to max-seq-length=1024 (memory-limited, ~15.6GB
    # cards), which truncates over half these examples. Front-loading the citation
    # means it survives truncation even when the detailed reasoning gets cut off --
    # exactly the information RQ2 (citation accuracy) actually measures.
    answer = f"Decision: {decision}."
    if rules:
        rule_list = ", ".join(sorted(set(rules), key=rules.index))
        answer += f" Rules cited: {edition} rule(s) {rule_list}."
    answer += f"\n\n{assessment}"
    return answer


def record_to_example(record):
    user_msg = build_user_message(record)
    assistant_msg = build_assistant_message(record)
    if not user_msg or not assistant_msg:
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        # kept for traceability back to the source ruling; mlx_lm ignores extra keys
        "source_url": record.get("source_url"),
        "group": record.get("group"),
    }


def load_examples():
    examples = []
    dropped_incomplete = 0
    for path in RAW_DIR.glob("asa_rulings_*.jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                example = record_to_example(record)
                if example is None:
                    dropped_incomplete += 1
                    continue
                examples.append(example)
    print(f"Built {len(examples)} examples, dropped {dropped_incomplete} (missing ad_description/issue or assessment)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    kept, dropped_too_long = [], 0
    for ex in examples:
        n_tokens = len(tokenizer.apply_chat_template(ex["messages"], return_dict=False))
        if n_tokens > MAX_TOKENS:
            dropped_too_long += 1
            continue
        kept.append(ex)
    print(f"Dropped {dropped_too_long} examples over {MAX_TOKENS} tokens (kept {len(kept)})")
    return kept


def split_and_write(examples):
    random.Random(SEED).shuffle(examples)
    n = len(examples)
    n_train = int(n * SPLIT_RATIOS[0])
    n_valid = int(n * SPLIT_RATIOS[1])

    splits = {
        "train": examples[:n_train],
        "valid": examples[n_train:n_train + n_valid],
        "test": examples[n_train + n_valid:],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        out_path = OUT_DIR / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(rows)} examples -> {out_path}")


if __name__ == "__main__":
    examples = load_examples()
    split_and_write(examples)
