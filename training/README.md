# Qwen LoRA Training For Noryx

Dataset used:

```python
from datasets import load_dataset

ds = load_dataset(
    "armand0e/claude-fable-5-claude-code",
    split="train",
    streaming=True,
)
```

This prepares the requested streaming dataset, fine-tunes a trainable Hugging Face
Qwen model with LoRA, then exports to GGUF for llama.cpp. RAG remains separate
and still runs first in the backend.

Commands:

```powershell
.\venv\Scripts\pip.exe install -r training\requirements-train.txt

.\venv\Scripts\python.exe -X utf8 training\prepare_claude_fable_dataset.py `
  --dataset armand0e/claude-fable-5-claude-code `
  --output training\data\claude_fable_messages.jsonl

.\venv\Scripts\python.exe training\train_claude_fable_lora.py `
  --base-model Qwen/Qwen2.5-3B-Instruct `
  --data-file training\data\claude_fable_messages.jsonl `
  --output-dir training\output\claude-fable-qwen-lora

.\venv\Scripts\python.exe training\merge_lora.py `
  --base-model Qwen/Qwen2.5-3B-Instruct `
  --adapter training\output\claude-fable-qwen-lora `
  --output-dir training\output\claude-fable-qwen-merged

powershell -ExecutionPolicy Bypass -File training\convert_to_gguf.ps1
```

After conversion, point llama.cpp at:

```text
models\qwen2.5-3b-claude-fable-f16.gguf
```

The existing PDF-first RAG flow stays unchanged:

```text
PDF QA lookup -> model answer -> "I don't have that information" fallback
```

Note: this machine currently reports CPU-only PyTorch. Qwen2.5-3B LoRA training
will be extremely slow without a CUDA GPU.
