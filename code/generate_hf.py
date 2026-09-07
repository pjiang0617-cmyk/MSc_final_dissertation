"""
Retrieve-then-generate for a single query.

python generate_hf.py "A weight-loss supplement ad claims users will lose 10lbs in one week." --adapter-path ../adapters/legal_lora_v1_gpu

Pass a plain description, not a full question -- see build_messages() for why.
"""

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from retrieve import load_index, retrieve_stratified, MODEL_NAME as EMBED_MODEL_NAME
from build_finetune_dataset import SYSTEM_PROMPT
from sentence_transformers import SentenceTransformer

LLM_MODEL_ID = "google/gemma-4-E4B-it"

# Strips a leading "Advertisement:" prefix if the caller included one, so it can't
# end up duplicated by build_messages() below.
LEADING_WRAPPER_RE = re.compile(r"^\s*advertisement\s*:\s*", re.IGNORECASE)


def clean_query(query):
    return LEADING_WRAPPER_RE.sub("", query).strip()


def format_context(results):
    return "\n\n".join(f"[{i}] {chunk['citation']}\n{chunk['text']}" for i, (score, chunk) in enumerate(results, 1))


def build_messages(query, results):
    # `query` is a plain, undecorated description of the ad/behaviour -- the same
    # text used for retrieval in main() below. The full "Advertisement: ... Does
    # this breach ...?" wrapper that build_finetune_dataset.py trains on is added
    # only here, for the LLM-facing prompt, not for the retrieval query.
    context = format_context(results)
    user_content = (
        f"Relevant rules and legislation:\n{context}\n\n"
        f"Advertisement:\n{query}\n\n"
        "Does this advertisement breach the CAP Code or BCAP Code? If so, which rule(s) and why?"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def load_model(adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    # device_map={"": 0}, not "auto": avoids accelerate's automatic CPU/disk offload,
    # which 4-bit bitsandbytes models reject.
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, quantization_config=quant_config, device_map={"": 0}, torch_dtype=torch.bfloat16)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k-rules", type=int, default=3, help="how many cap_rule/bcap_rule/legislation_section chunks to retrieve")
    ap.add_argument("--k-cases", type=int, default=2, help="how many asa_case_summary/asa_case_assessment chunks to retrieve")
    ap.add_argument("--group", default=None)
    ap.add_argument("--adapter-path", default=None, help="path to a trained LoRA adapter (fine-tuned model)")
    ap.add_argument("--max-tokens", type=int, default=500)
    args = ap.parse_args()
    query = clean_query(args.query)

    print("Loading retrieval index...")
    chunks, embeddings = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    results = retrieve_stratified(query, chunks, embeddings, embed_model, k_rules=args.k_rules, k_cases=args.k_cases, group=args.group)

    print(f"Retrieved {len(results)} chunks:")
    for score, chunk in results:
        print(f"  [{score:.3f}] {chunk['citation']}")

    print("\nLoading generation model...")
    model, tokenizer = load_model(args.adapter_path)
    messages = build_messages(query, results)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("\n--- Answer ---")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
    answer = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(answer)
    return answer


if __name__ == "__main__":
    main()
