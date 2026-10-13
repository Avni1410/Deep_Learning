"""
Phase 8 - hard-negative audit, adapted directly from the real
scripts/audit_phase6_hard_negatives.py (reused methodology, not reinvented),
pointed at Phase 8's semi-hard mined CSVs instead.

The key number to check against Phase 6's own audit:
  Phase 6: negative_1 > positive in 95.7-96.9% of queries (the diagnosed
           cause of Phase 6's regression).
  Phase 8: this should be 0% BY CONSTRUCTION, since semi-hard mining only
           accepts a negative if its similarity is strictly below the
           positive's. If this audit reports anything other than 0% here,
           that means the semi-hard filter has a bug - investigate before
           trusting any training run's results.

Also reports the `shortfall` column (queries where fewer than n_hard
semi-hard candidates existed) - this is an expected, honestly-reported
trade-off of the fix, not a defect: a strict safety guarantee (negatives
never outrank the positive) necessarily costs some negatives on queries
where the model already ranks almost everything below the positive.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def audit_file(category: str, csv_path: Path) -> dict | None:
    print("\n" + "=" * 70)
    print(f"AUDITING (Phase 8, semi-hard): {category.upper()}")
    print("=" * 70)

    if not csv_path.exists():
        print(f"ERROR: CSV not found:\n{csv_path}")
        return None

    df = pd.read_csv(csv_path)
    print(f"CSV: {csv_path}")
    print(f"Queries: {len(df)}")

    # --- target/candidate leakage (should always be 0, same guarantee as Phase 6) ---
    leakage_rows = []
    for _, row in df.iterrows():
        forbidden = {str(row["target"]), str(row["candidate"])}
        negs = [str(row.get(f"negative_{i}", "")) for i in (1, 2, 3) if str(row.get(f"negative_{i}", ""))]
        leaked = [n for n in negs if n in forbidden]
        if leaked:
            leakage_rows.append((row["query_id"], leaked))
    print(f"Target/reference leakage: {len(leakage_rows)}")

    # --- shortfall distribution (expected, honest trade-off) ---
    shortfall = pd.to_numeric(df.get("shortfall", pd.Series([0] * len(df))), errors="coerce").fillna(0)
    n_with_shortfall = int((shortfall > 0).sum())
    print(f"Queries with fewer than n_hard semi-hard candidates (shortfall > 0): "
          f"{n_with_shortfall}/{len(df)} ({100 * n_with_shortfall / max(1, len(df)):.2f}%)")

    # --- the critical check: does ANY negative still beat the positive? ---
    positive = pd.to_numeric(df.get("positive_similarity"), errors="coerce")
    sim_cols = [c for c in ("similarity_1", "similarity_2", "similarity_3") if c in df.columns]
    violations = 0
    for col in sim_cols:
        neg_sim = pd.to_numeric(df[col], errors="coerce")
        valid = positive.notna() & neg_sim.notna()
        violations += int((neg_sim[valid] >= positive[valid]).sum())
    print(f"Semi-hard GUARANTEE VIOLATIONS (any negative >= positive): {violations} "
          f"{'  <-- SHOULD BE 0 - investigate if not' if violations else '  (confirmed: 0, as expected)'}")

    valid = positive.notna()
    if valid.any() and "similarity_1" in df.columns:
        neg1 = pd.to_numeric(df["similarity_1"], errors="coerce")
        v = valid & neg1.notna()
        if v.any():
            margin = positive[v] - neg1[v]
            print(f"Mean positive similarity: {positive[v].mean():.6f}")
            print(f"Mean negative-1 similarity: {neg1[v].mean():.6f}")
            print(f"Mean margin (positive - negative-1): {margin.mean():.6f}  "
                  f"(compare against Phase 6's -0.07 to -0.08 - this should now be POSITIVE)")

    passed = len(leakage_rows) == 0 and violations == 0
    print(f"\nAUDIT STATUS: {'PASS' if passed else 'FAIL'}")
    return {
        "category": category, "queries": len(df), "target_reference_leakage": len(leakage_rows),
        "shortfall_queries": n_with_shortfall, "guarantee_violations": violations,
        "mean_margin": float(margin.mean()) if valid.any() and v.any() else float("nan"),
        "audit_pass": passed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-negative-dir", type=Path, required=True,
                    help="e.g. experiments/phase8/<run-name>/hard_negatives, for one specific round")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--categories", nargs="+", default=["dress", "shirt", "toptee"])
    args = ap.parse_args()

    print("=" * 70)
    print(f"PHASE 8 SEMI-HARD NEGATIVE AUDIT - round {args.round}")
    print("=" * 70)

    results = []
    for category in args.categories:
        csv_path = args.hard_negative_dir / f"round{args.round}_{category}_negatives.csv"
        result = audit_file(category, csv_path)
        if result is not None:
            results.append(result)

    if not results:
        print("\nNo CSV files were audited.")
        return

    summary = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    all_pass = bool(summary["audit_pass"].all())
    print("\n" + ("OVERALL AUDIT: PASS" if all_pass else "OVERALL AUDIT: FAIL - investigate before training on this round's negatives"))


if __name__ == "__main__":
    main()
