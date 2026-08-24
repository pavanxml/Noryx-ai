"""LoRA fine-tuning pipeline for armand0e/claude-fable-5-claude-code.

This trains a LoRA adapter on a trainable Hugging Face Qwen model. It does not
modify GGUF files directly. After training, merge/export and convert to GGUF for
llama.cpp.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


DEFAULT_DATASET = "armand0e/claude-fable-5-claude-code"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def _first_existing(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = row.get(name)
        if value:
            return value
    return None


def format_example(row: dict[str, Any], tokenizer) -> str:
    messages = _first_existing(row, ("messages", "conversation", "conversations"))
    if isinstance(messages, list):
        normalized = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or msg.get("from") or "user"
            content = msg.get("content") or msg.get("value") or msg.get("text") or ""
            role = {"human": "user", "gpt": "assistant", "claude": "assistant"}.get(str(role).lower(), role)
            if content:
                normalized.append({"role": role, "content": str(content)})
        if normalized:
            return tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=False)

    prompt = _first_existing(row, ("prompt", "instruction", "input", "question"))
    response = _first_existing(row, ("response", "output", "answer", "completion"))
    if prompt and response:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": str(response)}],
            tokenize=False,
            add_generation_prompt=False,
        )

    text = _first_existing(row, ("text", "content"))
    if text:
        return str(text)

    return json.dumps(row, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--data-file", default="training/data/claude_fable_messages.jsonl")
    parser.add_argument("--base-model", default=os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--output-dir", default="training/output/claude-fable-qwen-lora")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=100)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.data_file and Path(args.data_file).exists():
        ds = load_dataset("json", data_files=args.data_file, split="train")
    else:
        ds = load_dataset(args.dataset, split=args.split)
    ds = ds.map(lambda row: {"text": format_example(row, tokenizer)}, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    train_args = SFTConfig(
        output_dir=str(out),
        max_seq_length=args.max_seq_length,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=False,
        fp16=True,
        dataset_text_field="text",
        packing=False,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"Saved LoRA adapter to {out}")


if __name__ == "__main__":
    main()

