"""
GPU/CUDA training script -- the HF PEFT + bitsandbytes equivalent of the Mac's
`mlx_lm.lora` command (that MLX version stays untouched in
asa_legal_pipeline/code/ -- this is a separate, from-scratch implementation for
NVIDIA hardware, not a port of it, since MLX doesn't run on CUDA at all).

Uses plain transformers.Trainer + manual PEFT wrapping, NOT trl's SFTTrainer/
SFTConfig -- that higher-level wrapper kept breaking across the trl version
actually installed on the cluster (1.9.2): DataCollatorForCompletionOnlyLM was
removed (replaced by an SFTConfig flag), SFTConfig.max_seq_length was renamed
to max_length, and assistant_only_loss required chat-template "{% generation %}"
markers this checkpoint's template doesn't have. Rather than keep chasing trl's
API across versions, this drops down to the stable, rarely-churning
transformers.Trainer API and does the same two things trl's abstractions were
doing for us, explicitly:
  1. PEFT-wrap the model ourselves (get_peft_model) instead of letting
     SFTTrainer do it internally
  2. mask the prompt (system+user) out of the loss ourselves, using the exact
     same technique as the Mac's --mask-prompt / mlx_lm's ChatDataset.process:
     tokenize messages[:-1] with add_generation_prompt=True to find where the
     assistant's answer starts, then set every label token before that index
     to -100 (ignored by the loss) -- see mask_labels() below

Mirrors the same design decisions as the MLX run otherwise:
  - QLoRA (4-bit quantization via bitsandbytes) so it fits on a single GPU,
    matching the project plan's stated approach ("Hugging Face PEFT library
    implementing LoRA for efficient parameter updates on a single consumer-grade GPU")
  - same LoRA rank/alpha/target-module choices as a reasonable default for a
    Gemma-family model; adjust --lora-r / --lora-alpha if the first run under-fits
  - target_modules is a regex scoped to model.language_model.* only -- this
    checkpoint is multimodal (text+vision+audio), and bare-name matching (e.g.
    "q_proj") also matches same-named layers inside the vision/audio towers,
    which use a custom Gemma4ClippableLinear wrapper PEFT can't inject a LoRA
    adapter into (confirmed on the real cluster). We only want the text path
    anyway -- fine-tuning here is about teaching citation *behaviour*, not
    vision/audio capability (see build_finetune_dataset.py's docstring).

Usage:
    python train_hf.py --output-dir ../adapters/legal_lora_v1_gpu
    python train_hf.py --output-dir ../adapters/legal_lora_v1_gpu --max-steps 20  # smoke test
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "finetune"
MODEL_ID = "google/gemma-4-E4B-it"

LORA_TARGET_MODULES = r"^model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def build_model_and_tokenizer(load_in_4bit=True):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if load_in_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        # device_map="auto" lets accelerate decide per-layer placement, and on this
        # cluster it was deciding to offload some layers to CPU/disk -- which 4-bit
        # bitsandbytes models don't support without an extra opt-in flag, so it just
        # errors out. We're always requesting a single GPU (-G1) via Slurm anyway, so
        # skip the auto-placement guesswork and force everything onto that one GPU.
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    return model, tokenizer


def tokenize_and_mask(example, tokenizer, max_length):
    """Same logic as mlx_lm's ChatDataset.process: tokenize the full conversation,
    separately tokenize everything except the final (assistant) message with
    add_generation_prompt=True to find the offset where the assistant's answer
    starts, then mask every label before that offset with -100 so the loss is
    only computed on the assistant's actual answer."""
    messages = example["messages"]
    # return_dict=False is required here -- without it, newer transformers versions
    # return a dict (like a BatchEncoding) instead of a plain list of token ids.
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, truncation=True, max_length=max_length)
    prompt_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, return_dict=False, add_generation_prompt=True, truncation=True, max_length=max_length)
    offset = min(len(prompt_ids), len(full_ids))

    labels = list(full_ids)
    for i in range(offset):
        labels[i] = -100

    return {"input_ids": full_ids, "labels": labels}


def make_collate_fn(pad_token_id):
    def collate(batch):
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, labels, attention_mask = [], [], []
        for ex in batch:
            ids, labs = ex["input_ids"], ex["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_token_id] * pad_len)
            labels.append(labs + [-100] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, help="where to save the LoRA adapter")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    # Started at 2, assuming a real GPU would comfortably beat the Mac's forced
    # batch-size=1 -- but the "cs" partition's cards only have ~15.6GB VRAM, and
    # without gradient checkpointing (now fixed above) even batch-size=2 OOM'd.
    # Back to 1 as the safe default; --grad-accum makes up the effective batch size.
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--num-epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="set e.g. 20 for a smoke test; -1 = full run over num-epochs")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--save-steps", type=int, default=20)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--resume-from-checkpoint", default=None)
    args = ap.parse_args()

    print("Loading model in 4-bit (QLoRA)...")
    model, tokenizer = build_model_and_tokenizer(load_in_4bit=True)

    # Manual replacement for peft's prepare_model_for_kbit_training. That helper
    # upcasts EVERY non-4bit param in the whole model to fp32 for numerical
    # stability -- fine for a text-only model, but this checkpoint is multimodal,
    # and its (unquantized, much larger than the text decoder) vision/audio towers
    # got upcast right along with it, which alone tried to allocate 10.5GB and
    # OOM'd on this 15.6GB GPU. We only ever train/use model.language_model, so
    # scope the fp32 upcast (and just do the rest of what the helper does --
    # freeze everything, enable gradient checkpointing + input-grad hook) to that.
    for name, param in model.named_parameters():
        param.requires_grad = False
        if param.ndim == 1 and "language_model" in name:
            param.data = param.data.to(torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Loading and tokenizing dataset...")
    dataset = load_dataset("json", data_files={
        "train": str(DATA_DIR / "train.jsonl"),
        "validation": str(DATA_DIR / "valid.jsonl"),
    })
    dataset = dataset.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, args.max_seq_length),
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=make_collate_fn(tokenizer.pad_token_id),
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    print(f"Saved final adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()
