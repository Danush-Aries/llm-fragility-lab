#!/usr/bin/env python3
"""
LLM Fragility Lab — CLI entry point.

Usage
-----
  python main.py                        # run built-in demo
  python main.py --response "..." --truth "..."   # single comparison
  python main.py --batch samples.json   # batch evaluation from a JSON file

JSON batch file format:
  [
    {"response": "...", "ground_truth": "..."},
    ...
  ]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.app.engines.hallucination_hunter import HallucinationHunter


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

DEMO_PAIRS = [
    (
        "The Eiffel Tower is located in Paris, France and was completed in 1889.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France, "
        "constructed between 1887 and 1889.",
    ),
    (
        "Python was created by Guido van Rossum and first released in 1991.",
        "Python is a high-level programming language created by Guido van Rossum, "
        "with its first version released in 1991.",
    ),
    (
        "The moon is made of green cheese and orbits the sun directly.",
        "The Moon is Earth's only natural satellite, orbiting Earth at an average "
        "distance of about 384,400 km.",
    ),
    (
        "Photosynthesis converts sunlight into chemical energy stored in glucose.",
        "Photosynthesis is the process by which plants use sunlight, water, and "
        "carbon dioxide to produce oxygen and energy in the form of glucose.",
    ),
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _print_result(idx: int, response: str, ground_truth: str, result: dict) -> None:
    risk_icons = {"low": "[OK ]", "medium": "[MED]", "high": "[HGH]"}
    icon = risk_icons[result["hallucination_risk"]]
    print(f"\n--- Sample {idx} {icon} ---")
    print(f"  Response   : {response[:80]}{'...' if len(response) > 80 else ''}")
    print(f"  Ground truth: {ground_truth[:80]}{'...' if len(ground_truth) > 80 else ''}")
    print(f"  Drift score : {result['drift_score']:.4f}  "
          f"(risk: {result['hallucination_risk']})")
    print(f"  Token F1    : {result['token_f1']:.4f}  "
          f"Jaccard: {result['jaccard_similarity']:.4f}  "
          f"Bigram: {result['bigram_overlap']:.4f}")
    print(f"  Novel tokens: {result['novel_token_ratio']:.4f}  "
          f"Length ratio: {result['length_ratio']:.4f}")


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Samples analysed : {summary['n']}")
    print(f"  Mean drift score : {summary['mean_drift_score']:.4f}")
    print(f"  Min  drift score : {summary['min_drift_score']:.4f}")
    print(f"  Max  drift score : {summary['max_drift_score']:.4f}")
    rc = summary["risk_counts"]
    print(f"  Risk distribution: low={rc['low']}  medium={rc['medium']}  high={rc['high']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM Fragility Lab — hallucination drift scorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--response", type=str, help="LLM response text (single mode)")
    p.add_argument("--truth", type=str, help="Ground truth text (single mode)")
    p.add_argument(
        "--batch",
        type=str,
        metavar="FILE",
        help="Path to a JSON file containing an array of {response, ground_truth} objects",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of formatted text",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    hunter = HallucinationHunter()

    # ---- Batch mode -------------------------------------------------------
    if args.batch:
        batch_path = Path(args.batch)
        if not batch_path.exists():
            print(f"ERROR: file not found: {args.batch}", file=sys.stderr)
            return 1
        with batch_path.open() as fh:
            data = json.load(fh)
        pairs = [(item["response"], item["ground_truth"]) for item in data]
        results = hunter.batch_analyze(pairs)
        summary = hunter.summary(results)

        if args.json:
            print(json.dumps({"results": results, "summary": summary}, indent=2))
        else:
            for i, (pair, result) in enumerate(zip(pairs, results), start=1):
                _print_result(i, pair[0], pair[1], result)
            _print_summary(summary)
        return 0

    # ---- Single mode -------------------------------------------------------
    if args.response or args.truth:
        if not (args.response and args.truth):
            print("ERROR: both --response and --truth are required in single mode.",
                  file=sys.stderr)
            return 1
        result = hunter.analyze(args.response, args.truth)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_result(1, args.response, args.truth, result)
        return 0

    # ---- Demo mode ---------------------------------------------------------
    print("LLM Fragility Lab — Demo Run")
    print("=" * 60)
    results = hunter.batch_analyze(DEMO_PAIRS)
    for i, (pair, result) in enumerate(zip(DEMO_PAIRS, results), start=1):
        _print_result(i, pair[0], pair[1], result)
    _print_summary(hunter.summary(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
