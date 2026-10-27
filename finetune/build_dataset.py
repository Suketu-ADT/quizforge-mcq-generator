"""Combine SciQ examples and teacher (distilled) examples into train.jsonl / val.jsonl.

    python -m finetune.build_dataset --sciq data/sciq_train.jsonl --distilled data/distilled.jsonl --out data --sciq-examples 1800

The teacher examples are repeated --distilled-weight times (default 2): there are far fewer of them, but they are the
closest to real app requests (longer passages, all difficulties, specific topic labels).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from finetune.prepare_data import build_sciq_examples, read_jsonl, split, summary, write_jsonl


def build(sciq_rows: list[dict], distilled: list[dict], sciq_examples: int, distilled_weight: int = 2, val_fraction: float = 0.05,
          seed: int = 7) -> tuple[list[dict], list[dict], dict]:
    sciq = build_sciq_examples(sciq_rows, sciq_examples, seed=seed)
    sciq_train, sciq_val = split(sciq, val_fraction, seed)
    dist_train, dist_val = split(distilled, val_fraction, seed) if distilled else ([], [])
    # validation keeps one copy of each example; repeats only weight the training set, and never cross the split
    train = sciq_train + dist_train * max(1, distilled_weight)
    val = sciq_val + dist_val
    info = {"train": summary(train), "val": summary(val), "sciq_examples": len(sciq), "distilled_examples": len(distilled),
            "distilled_weight": distilled_weight, "seed": seed}
    return train, val, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sciq", type=Path, required=True, help="SciQ training rows (JSON lines)")
    ap.add_argument("--distilled", type=Path, help="teacher examples from finetune.distill (optional)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sciq-examples", type=int, default=1800)
    ap.add_argument("--distilled-weight", type=int, default=2)
    args = ap.parse_args(argv)
    distilled = read_jsonl(args.distilled) if args.distilled and args.distilled.exists() else []
    train, val, info = build(read_jsonl(args.sciq), distilled, args.sciq_examples, args.distilled_weight)
    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "val.jsonl", val)
    (args.out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
