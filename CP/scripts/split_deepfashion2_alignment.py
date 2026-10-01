"""
Phase 5 - deterministic, leakage-free split of the DeepFashion2 5K alignment subset.

Uses union-find: any two pairs sharing a user_image OR a shop_image are
merged into one connected component. Whole components are assigned to
train/val, never split - this makes leakage structurally impossible,
rather than patched after the fact.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=REPO_ROOT / "data/processed/deepfashion2/domain_alignment_pairs_5000.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/processed/deepfashion2")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    n = len(df)
    print(f"Loaded {n} pairs. Columns: {list(df.columns)}")

    uf = UnionFind(n)
    first_seen: dict[str, int] = {}
    for col in ("user_image", "shop_image"):
        for i, val in enumerate(df[col]):
            if val in first_seen:
                uf.union(i, first_seen[val])
            else:
                first_seen[val] = i

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        components[uf.find(i)].append(i)
    comp_list = list(components.values())
    print(f"Connected components: {len(comp_list)} | largest component: {max(len(c) for c in comp_list)} pairs")

    rng = random.Random(args.seed)
    rng.shuffle(comp_list)

    val_target = int(n * args.val_fraction)
    val_indices: list[int] = []
    train_indices: list[int] = []
    for comp in comp_list:
        if len(val_indices) < val_target:
            val_indices.extend(comp)
        else:
            train_indices.extend(comp)

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)

    user_overlap = set(train_df["user_image"]) & set(val_df["user_image"])
    shop_overlap = set(train_df["shop_image"]) & set(val_df["shop_image"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "alignment_train.csv"
    val_path = args.out_dir / "alignment_val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Train pairs: {len(train_df)} ({len(train_df)/n:.1%}) | "
          f"Val pairs: {len(val_df)} ({len(val_df)/n:.1%})")
    print(f"User overlap: {len(user_overlap)} (must be 0) | Shop overlap: {len(shop_overlap)} (must be 0)")
    assert not user_overlap and not shop_overlap, "Leakage remains - investigate before proceeding."
    print(f"Saved {train_path}\nSaved {val_path}")


if __name__ == "__main__":
    main()