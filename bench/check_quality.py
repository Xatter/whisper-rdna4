#!/usr/bin/env python3
"""WER of our transcript vs a baseline transcript (DESIGN.md S8.3).

Simple normalized-word WER: lowercase, strip punctuation, collapse
whitespace, split on whitespace, then standard Levenshtein edit distance
over the word sequences, normalized by len(reference_words).

Usage:
    check_quality.py --ours ours.txt --baseline bench-results/44e8...json
    check_quality.py --ours ours.txt --baseline-text baseline.txt

The baseline JSON is insanely-fast-whisper-rocm's export format
(bench-results/*.json on gpu-host) -- this script extracts its "text"
field (or concatenates "chunks[].text" if "text" is absent).
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> list[str]:
    text = text.lower().translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ") if text else []


def word_edit_distance(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    # rolling-row DP, O(n*m) time, O(m) space
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,  # deletion
                cur[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = cur
    return prev[m]


def wer(reference: str, hypothesis: str) -> dict:
    ref_words = normalize(reference)
    hyp_words = normalize(hypothesis)
    if not ref_words:
        return {"wer": float("nan"), "edits": 0, "ref_words": 0, "hyp_words": len(hyp_words)}
    edits = word_edit_distance(ref_words, hyp_words)
    return {
        "wer": edits / len(ref_words),
        "edits": edits,
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
    }


def load_baseline_text(path: str) -> str:
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "text" in data and data["text"]:
            return data["text"]
        chunks = data.get("chunks") if isinstance(data, dict) else None
        if chunks:
            return " ".join(c.get("text", "") for c in chunks)
        raise ValueError(f"couldn't find transcript text in {path}")
    with open(path) as f:
        return f.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="path to our transcript (.txt or .json)")
    ap.add_argument("--baseline", required=True, help="path to baseline transcript (.txt or .json)")
    ap.add_argument("--max-wer", type=float, default=0.03, help="fail if WER exceeds this")
    args = ap.parse_args()

    ours_text = load_baseline_text(args.ours)
    baseline_text = load_baseline_text(args.baseline)

    result = wer(baseline_text, ours_text)
    print(json.dumps(result, indent=2))

    if result["wer"] != result["wer"]:  # NaN
        print("WARNING: empty reference text, WER undefined", file=sys.stderr)
        return 1
    if result["wer"] > args.max_wer:
        print(
            f"FAIL: WER {result['wer']:.4f} exceeds threshold {args.max_wer}",
            file=sys.stderr,
        )
        return 1
    print(f"OK: WER {result['wer']:.4f} <= {args.max_wer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
