"""
GPU/CUDA variant of asa_legal_pipeline/code/generate.py -- same retrieve-then-generate
logic, but the generation half uses transformers+peft instead of mlx_lm (which only
runs on Apple Silicon). retrieve.py itself is shared unchanged (sentence-transformers
runs fine on either CPU or CUDA).

Usage:
    python generate_hf.py "does this ad breach the CAP Code if it claims a specific weight loss in a week?"
    python generate_hf.py "..." --adapter-path ../adapters/legal_lora_v1_gpu   # fine-tuned model
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from retrieve import load_index, retrieve, MODEL_NAME as EMBED_MODEL_NAME
from build_finetune_dataset import SYSTEM_PROMPT
from sentence_transformers import SentenceTransformer

LLM_MODEL_ID = "google/gemma-4-E4B-it"


def format_context(results):
    return "\n\n".join(f"[{i}] {chunk['citation']}\n{chunk['text']}" for i, (score, chunk) in enumerate(results, 1))


def build_messages(query, results):
    # Reuses the exact system prompt and user-message shape build_finetune_dataset.py
    # trains on (an "Advertisement: ... Does this breach ...?" block) -- only the
    # retrieved context is new. A fine-tuned adapter learned its "Decision: ...
    # Rules cited: ..." habit against that shape; feeding it an unrelated QA-style
    # prompt (the earlier version of this script) would give it an input format it
    # never saw during training, making any RAG-vs-no-RAG comparison meaningless.
    context = format_context(results)
    user_content = f"Relevant rules and legislation:\n{context}\n\n{query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def load_model(adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    # device_map={"": 0} not "auto" -- see train_hf.py's comment: "auto" was offloading
    # layers to CPU on this cluster, which 4-bit bitsandbytes models reject outright.
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, quantization_config=quant_config, device_map={"": 0}, torch_dtype=torch.bfloat16)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--group", default=None)
    ap.add_argument("--source-type", default=None)
    ap.add_argument("--adapter-path", default=None, help="path to a trained LoRA adapter (fine-tuned model)")
    ap.add_argument("--max-tokens", type=int, default=500)
    args = ap.parse_args()

    print("Loading retrieval index...")
    chunks, embeddings = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    results = retrieve(args.query, chunks, embeddings, embed_model, k=args.k, group=args.group, source_type=args.source_type)

    print(f"Retrieved {len(results)} chunks:")
    for score, chunk in results:
        print(f"  [{score:.3f}] {chunk['citation']}")

    print("\nLoading generation model...")
    model, tokenizer = load_model(args.adapter_path)
    messages = build_messages(args.query, results)
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
