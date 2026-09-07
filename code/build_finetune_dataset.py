"""
Turns the raw ASA ruling records (data/raw/asa_rulings_*.jsonl) into
instruction-tuning pairs in the {"messages": [...]} chat format for LoRA
fine-tuning.

Design: the fine-tuned adapter is meant to teach *behaviour* (how to reason
from given ad copy to a rule-grounded verdict, and to always cite the exact
rule numbers), not to memorise facts -- the facts (rule text, statute text)
belong in the RAG index instead.

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
MODEL_DIR = "google/gemma-4-E4B-it"  # HF Hub id -- tokenizer/chat template only

# Examples longer than this (measured with the real tokenizer) are dropped from
# the dataset entirely, rather than relying on truncation at training time.
# Chosen to maximise training data: of the 534 examples with all required
# fields, only 16 exceed this length. Training's actual --max-seq-length is
# 1024 (see train_job.sbatch), well below this, so most examples between 1024
# and 3072 tokens are still truncated during training -- the front-loaded
# "Decision: ... Rules cited: ..." is designed to survive that; the detailed
# reasoning after it may not.
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

# The raw rulings are ~90% "Upheld", so without correction the model can reach
# a high decision-match score just by always predicting the majority class.
# Minority-decision training rows are duplicated (train split only -- valid/test
# stay at the natural distribution) until they reach roughly this share.
MINORITY_TARGET_RATIO = 0.3


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

    # "Rules cited:" comes right after "Decision:", before the detailed reasoning,
    # so it survives truncation at training/generation time even when the reasoning
    # that follows gets cut off.
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
        # kept for traceability back to the source ruling
        "source_url": record.get("source_url"),
        "group": record.get("group"),
        # same fallback as build_assistant_message(), so this matches what the
        # assistant message actually says
        "decision": record.get("decision") or "Upheld",
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


def oversample_minority(rows, target_ratio=MINORITY_TARGET_RATIO, seed=SEED):
    from collections import Counter

    counts = Counter(r["decision"] for r in rows)
    if len(counts) < 2:
        return rows
    majority_label, majority_count = counts.most_common(1)[0]

    oversampled = list(rows)
    for label, count in counts.items():
        if label == majority_label:
            continue
        target_count = int(target_ratio * majority_count / (1 - target_ratio))
        factor = max(1, round(target_count / count))
        if factor > 1:
            oversampled.extend([r for r in rows if r["decision"] == label] * (factor - 1))

    random.Random(seed).shuffle(oversampled)
    return oversampled


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

    from collections import Counter
    print(f"  train decision counts before oversampling: {dict(Counter(r['decision'] for r in splits['train']))}")
    splits["train"] = oversample_minority(splits["train"])
    print(f"  train decision counts after oversampling:  {dict(Counter(r['decision'] for r in splits['train']))}")

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
