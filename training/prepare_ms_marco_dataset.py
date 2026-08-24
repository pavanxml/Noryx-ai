"""Prepare MS MARCO v1.1/v2.1 as chat-style instruction data.

Each row becomes a simple QA training example:
user: answer the query using the provided passages
assistant: answer from MS MARCO answer/wellFormedAnswers
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return [str(value)] if str(value).strip() else []


def _passage_texts(passages: Any, limit: int) -> list[str]:
    if not isinstance(passages, dict):
        return []
    texts = passages.get("passage_text") or passages.get("text") or []
    selected = passages.get("is_selected") or []
    rows: list[tuple[int, str]] = []
    for idx, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            continue
        score = 1 if idx < len(selected) and selected[idx] else 0
        rows.append((score, text.strip()))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in rows[:limit]]


def row_to_messages(row: dict[str, Any], passage_limit: int) -> dict[str, Any] | None:
    query = str(row.get("query") or "").strip()
    answers = _texts(row.get("wellFormedAnswers")) or _texts(row.get("answers"))
    answers = [a for a in answers if a.lower() != "no answer present."]
    if not query or not answers:
        return None
    passages = _passage_texts(row.get("passages"), passage_limit)
    passage_block = "\n".join(f"- {p}" for p in passages)
    user = f"Answer this question clearly and briefly.\n\nQuestion: {query}"
    if passage_block:
        user += f"\n\nRelevant passages:\n{passage_block}"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answers[0]},
        ],
        "source": "microsoft/ms_marco",
        "query_id": row.get("query_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["v1.1", "v2.1"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--passage-limit", type=int, default=3)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("microsoft/ms_marco", args.config, split=args.split)
    count = 0
    scanned = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            scanned += 1
            item = row_to_messages(row, args.passage_limit)
            if item:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
            if args.limit and scanned >= args.limit:
                break
    print(f"Wrote {count} examples to {out} from {scanned} scanned rows")


if __name__ == "__main__":
    main()
