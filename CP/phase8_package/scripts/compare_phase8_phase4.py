"""
Phase 8 vs Phase 4 comparison.

Reads Phase 8's experiments/phase8/results.json (produced by
scripts/evaluate_baseline.py run against the Phase 8 ranked files) and
prints it next to the FIXED, already-validated Phase 4 numbers. Phase 4's
numbers are hardcoded here deliberately - they are a frozen historical
result (experiments/phase4/results.json), not something this script should
ever recompute or risk silently drifting from.

Usage:
  python scripts/evaluate_baseline.py --ranked \
      experiments/phase8/results/dress_val_phase8_ranked.json \
      experiments/phase8/results/shirt_val_phase8_ranked.json \
      experiments/phase8/results/toptee_val_phase8_ranked.json \
      --out experiments/phase8/results.json
  python scripts/compare_phase8_phase4.py --phase8-results experiments/phase8/results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PHASE4_BASELINE = {
    "dress":   {"R@1": 0.0398, "R@5": 0.1306, "R@10": 0.2047, "R@50": 0.4643},
    "shirt":   {"R@1": 0.0567, "R@5": 0.1664, "R@10": 0.2427, "R@50": 0.5219},
    "toptee":  {"R@1": 0.0745, "R@5": 0.2196, "R@10": 0.3171, "R@50": 0.5892},
    "overall": {"R@1": 0.0570, "R@5": 0.1722, "R@10": 0.2547, "R@50": 0.5251},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase8-results", type=Path, required=True,
                    help="output of evaluate_baseline.py run on the Phase 8 ranked files")
    args = ap.parse_args()

    with open(args.phase8_results, "r", encoding="utf-8") as f:
        data = json.load(f)
    entry = data[-1] if isinstance(data, list) else data  # last appended entry = most recent run
    p8_metrics = entry["metrics"]

    print("=" * 70)
    print("Phase 4 (validated baseline) vs Phase 8 (semi-hard, iterative re-mining)")
    print("=" * 70)
    header = f"{'Category':<10} {'Metric':<6} {'Phase 4':>10} {'Phase 8':>10} {'Delta':>10}"
    print(header)
    print("-" * len(header))
    for cat in ["dress", "shirt", "toptee", "overall"]:
        if cat not in p8_metrics:
            continue
        for k in ["R@1", "R@5", "R@10", "R@50"]:
            p4 = PHASE4_BASELINE[cat][k]
            p8 = p8_metrics[cat].get(k)
            if p8 is None:
                continue
            delta = p8 - p4
            flag = "  <-- WORSE" if delta < -0.001 else ("  <-- better" if delta > 0.001 else "")
            print(f"{cat:<10} {k:<6} {p4:>10.4f} {p8:>10.4f} {delta:>+10.4f}{flag}")
    print("=" * 70)
    print("Do not declare success from this table alone without checking whether")
    print("any 'better' deltas are consistent across R@1 AND R@10 (not just one K),")
    print("and whether the margin / hard-negative quality stats also improved -")
    print("see docs/phase_8_semi_hard_mining.md for the full required analysis.")


if __name__ == "__main__":
    main()
