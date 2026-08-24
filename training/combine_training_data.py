"""Combine prepared chat JSONL files into one training file."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="training/data/combined_training_messages.jsonl")
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w", encoding="utf-8") as wf:
        for raw in args.inputs:
            path = Path(raw)
            with path.open("r", encoding="utf-8") as rf:
                for line in rf:
                    if line.strip():
                        wf.write(line)
                        total += 1
    print(f"Wrote {total} combined examples to {out}")


if __name__ == "__main__":
    main()
