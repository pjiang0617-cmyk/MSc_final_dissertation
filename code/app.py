"""Web demo: a single text box wrapping the retrieve-then-generate pipeline,
showing the answer alongside the retrieved sources.

    python app.py                          # v2 adapter, port 7860
    python app.py --port 7861
    python app.py --adapter-path ../adapters/legal_lora_v1_gpu_bf16
"""

import argparse
import os

# Must be set before `import gradio` -- the compute node has no outbound internet.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from retrieve import load_index, retrieve_stratified, MODEL_NAME as EMBED_MODEL_NAME
from generate_hf import load_model, build_messages, clean_query
from sentence_transformers import SentenceTransformer
import torch

# v2, not v1: v1 predates the class-imbalance correction and answers "Upheld"
# for almost any input, so it is the wrong model to put in front of a tester.
DEFAULT_ADAPTER = "../adapters/legal_lora_v2_gpu_bf16"

ap = argparse.ArgumentParser()
ap.add_argument("--adapter-path", default=DEFAULT_ADAPTER)
ap.add_argument("--port", type=int, default=7860)
args = ap.parse_args()

print("Loading retrieval index...")
chunks, embeddings = load_index()
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print(f"Loading generation model with adapter: {args.adapter_path}")
model, tokenizer = load_model(adapter_path=args.adapter_path)

print("Ready")

def answer(query):
    q = clean_query(query)
    results = retrieve_stratified(q, chunks, embeddings, embed_model, k_rules=3, k_cases=2)

    sources_text = "\n\n".join(
        f"[{i}] {chunk['citation']}\n{chunk['text']}" for i, (score, chunk) in enumerate(results, 1)
    )
    messages = build_messages(q, results)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt,return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=500, do_sample=False)

    answer_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer_text, sources_text

import gradio as gr

demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(label="Describe the ad or behavior", lines=3),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Retrieved sources"),
    ],
    title="UK Advertising Compliance Assistant",
)

if __name__ == "__main__":
    # server_name="0.0.0.0": lets an SSH port forward from the login node reach
    # this port on the compute node.
    demo.launch(server_name="0.0.0.0", server_port=args.port)