"""Merge a trained LoRA adapter into a Hugging Face model directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default="training/output/claude-fable-qwen-lora")
    parser.add_argument("--output-dir", default="training/output/claude-fable-qwen-merged")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    print(f"Saved merged model to {out}")


if __name__ == "__main__":
    main()
