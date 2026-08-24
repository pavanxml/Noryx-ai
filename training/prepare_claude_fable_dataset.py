"""Stream armand0e/claude-fable-5-claude-code and write local chat JSONL.

The source dataset contains full Claude Code traces. For model tuning we keep the
actual conversational messages and drop tool metadata-heavy fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

SOURCE_DATASET = "armand0e/claude-fable-5-claude-code"
ROLE_MAP = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "claude": "assistant",
    "system": "system",
    "tool": "tool",
}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p for p in parts if p).strip()
    if content is None:
        return ""
    return str(content)


def normalize_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    normalized: list[dict[str, str]] = []
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            raw_role = str(msg.get("role") or msg.get("from") or "user").lower()
            role = ROLE_MAP.get(raw_role, "user")
            content = _message_text(msg.get("content") or msg.get("value") or msg.get("text"))
            if not content:
                continue
            if role == "tool":
                role = "user"
                content = "Tool result:\n" + content
            if role not in {"system", "user", "assistant"}:
                role = "user"
            normalized.append({"role": role, "content": content})
    if normalized:
        return normalized

    prompt = _message_text(row.get("prompt"))
    if prompt:
        return [{"role": "user", "content": prompt}]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=SOURCE_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="training/data/claude_fable_messages.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            messages = normalize_messages(row)
            if not messages:
                continue
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break
    print(f"Wrote {count} examples to {out}")


if __name__ == "__main__":
    main()

