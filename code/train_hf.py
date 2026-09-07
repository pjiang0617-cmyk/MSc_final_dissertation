"""
QLoRA fine-tuning script using Hugging Face transformers + peft + bitsandbytes.

Uses plain transformers.Trainer with manual PEFT wrapping, rather than trl's
SFTTrainer, and does explicitly what that higher-level wrapper would otherwise
do internally:
  1. PEFT-wrap the model (get_peft_model) before handing it to Trainer
  2. mask the prompt (system+user) out of the loss: tokenize messages[:-1] with
     add_generation_prompt=True to find where the assistant's answer starts,
     then set every label token before that index to -100 (ignored by the
     loss) -- see tokenize_and_mask() below

Key design points:
  - QLoRA (4-bit quantization via bitsandbytes) so the model fits on a single GPU
  - target_modules is a regex scoped to model.language_model.* only -- the base
    checkpoint is multimodal (text+vision+audio), and bare-name matching (e.g.
    "q_proj") would also match same-named layers inside the vision/audio
    towers, which use a custom linear layer implementation peft cannot inject
    a LoRA adapter into. Only the text path is trained anyway, since
    fine-tuning here is about citation *behaviour*, not vision/audio capability.

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
        # device_map={"": 0}, not "auto": a single GPU is always requested via Slurm
        # anyway, and 4-bit bitsandbytes models reject accelerate's automatic
        # CPU/disk offload without an extra opt-in flag.
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
    # batch-size=1 is the limit on ~15.6GB cards; --grad-accum makes up the
    # effective batch size instead.
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

    # Manual replacement for peft's prepare_model_for_kbit_training(): that helper
    # upcasts every non-4bit param in the whole model to fp32, which for this
    # multimodal checkpoint includes the much larger, unquantized vision/audio
    # towers and OOMs on a 15.6GB GPU. Only model.language_model is trained, so
    # the fp32 upcast is scoped to that; the rest (freeze everything, enable
    # gradient checkpointing + input-grad hook) mirrors what the helper does.
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
