"""
GPU/CUDA variant of asa_legal_pipeline/code/evaluate_finetune.py -- identical
metrics and scoring logic (score_answer, load_valid_rule_numbers, load_ground_truth_by_url
are unchanged, pure Python), only the model-loading/generation half is swapped from
mlx_lm to transformers+peft.

Usage:
    python evaluate_finetune_hf.py --adapter-path ../adapters/legal_lora_v1_gpu
    python evaluate_finetune_hf.py --adapter-path ../adapters/legal_lora_v1_gpu --limit 10
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from build_finetune_dataset import SYSTEM_PROMPT

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FINETUNE_DIR = Path(__file__).resolve().parent.parent / "data" / "finetune"
MODEL_ID = "google/gemma-4-E4B-it"

RULE_PATTERN = re.compile(r"\b\d{1,2}\.\d{1,3}(?:\.\d{1,3})?\b")


def load_valid_rule_numbers():
    valid = set()
    for fname in ("cap_code_sections.json", "bcap_code_sections.json"):
        sections = json.loads((RAW_DIR / fname).read_text(encoding="utf-8"))
        for sec in sections:
            for rule in sec["rules"]:
                valid.add(rule["rule_number"])
    return valid


def load_ground_truth_by_url():
    gt = {}
    for fname in ("asa_rulings_group_b.jsonl", "asa_rulings_group_d.jsonl"):
        with open(RAW_DIR / fname, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                gt[rec["source_url"]] = {
                    "decision": rec.get("decision"),
                    "cited_rules": set(rec.get("cited_rules") or []),
                }
    return gt


def extract_decision(text):
    m = re.search(r"Decision:\s*(Upheld|Not upheld|Partially upheld)", text, re.I)
    return m.group(1) if m else None


def extract_cited_rules(text):
    m = re.search(r"Rules cited:(.*)", text, re.I | re.S)
    scope = m.group(1) if m else text
    return set(RULE_PATTERN.findall(scope))


def score_answer(generated_text, ground_truth, valid_rules):
    pred_rules = extract_cited_rules(generated_text)
    true_rules = ground_truth["cited_rules"]

    if pred_rules:
        precision = len(pred_rules & true_rules) / len(pred_rules)
    else:
        precision = 0.0 if true_rules else 1.0
    recall = len(pred_rules & true_rules) / len(true_rules) if true_rules else 1.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    fabricated = pred_rules - valid_rules
    fabricated_rate = len(fabricated) / len(pred_rules) if pred_rules else 0.0

    pred_decision = extract_decision(generated_text)
    decision_match = (pred_decision is not None) and (pred_decision.lower() == (ground_truth["decision"] or "").lower())

    format_ok = bool(re.search(r"Decision:", generated_text, re.I)) and bool(re.search(r"Rules cited:", generated_text, re.I))

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "fabricated_rate": fabricated_rate, "fabricated_rules": sorted(fabricated),
        "decision_match": decision_match, "format_ok": format_ok,
    }


def load_model(adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    # device_map={"": 0} not "auto" -- see train_hf.py's comment: "auto" was offloading
    # layers to CPU on this cluster, which 4-bit bitsandbytes models reject outright.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config, device_map={"": 0}, torch_dtype=torch.bfloat16)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def run_model(model, tokenizer, user_content, max_tokens=500):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def summarize(name, results):
    n = len(results)
    avg = lambda key: sum(r[key] for r in results) / n if n else 0.0
    print(f"\n=== {name} (n={n}) ===")
    print(f"  rule citation precision: {avg('precision'):.3f}")
    print(f"  rule citation recall:    {avg('recall'):.3f}")
    print(f"  rule citation F1:        {avg('f1'):.3f}")
    print(f"  fabricated rule rate:    {avg('fabricated_rate'):.3f}")
    print(f"  decision match rate:     {avg('decision_match'):.3f}")
    print(f"  format adherence rate:   {avg('format_ok'):.3f}")


def evaluate_model(model, tokenizer, test_examples, ground_truth, valid_rules, max_tokens):
    results = []
    for i, ex in enumerate(test_examples):
        gt = ground_truth.get(ex["source_url"])
        if gt is None:
            continue
        user_content = ex["messages"][1]["content"]
        answer = run_model(model, tokenizer, user_content, max_tokens)
        results.append(score_answer(answer, gt, valid_rules))
        print(f"  [{i + 1}/{len(test_examples)}] {ex['source_url']}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=500)
    args = ap.parse_args()

    valid_rules = load_valid_rule_numbers()
    ground_truth = load_ground_truth_by_url()
    print(f"Loaded {len(valid_rules)} real CAP/BCAP rule numbers, {len(ground_truth)} ground-truth rulings")

    test_examples = [json.loads(l) for l in open(FINETUNE_DIR / "test.jsonl", encoding="utf-8")]
    if args.limit:
        test_examples = test_examples[:args.limit]
    print(f"Evaluating on {len(test_examples)} held-out test examples\n")

    # Loading both models concurrently OOMs on a 15.6GB card (base model alone uses
    # ~9GB) -- evaluate and fully release one model before loading the other.
    print("Loading base model...")
    base_model, base_tok = load_model(adapter_path=None)
    print("Evaluating base model...")
    base_results = evaluate_model(base_model, base_tok, test_examples, ground_truth, valid_rules, args.max_tokens)
    del base_model, base_tok
    torch.cuda.empty_cache()
    # Printed immediately (not held until the fine-tuned pass also finishes) so these
    # numbers survive even if the job is killed (e.g. hits its --time limit) partway
    # through the fine-tuned model's pass below.
    summarize("Baseline (no fine-tuning)", base_results)

    print("Loading fine-tuned model (adapter applied)...")
    ft_model, ft_tok = load_model(adapter_path=args.adapter_path)
    print("Evaluating fine-tuned model...")
    ft_results = evaluate_model(ft_model, ft_tok, test_examples, ground_truth, valid_rules, args.max_tokens)
    del ft_model, ft_tok
    torch.cuda.empty_cache()

    summarize("Fine-tuned", ft_results)


if __name__ == "__main__":
    main()
