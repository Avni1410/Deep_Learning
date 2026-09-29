"""
Phase 3 - retrieval evaluation metrics (Recall@K).

Supports one or more valid target ids per query, even though the current
FashionIQ CSVs provide exactly one target per row (see the docstring on
rank_of_target in cosine_retrieval.py for the "first match" semantics).
"""
from __future__ import annotations

from collections import defaultdict


def recall_at_k(ranks: list[int | None], k_values: list[int]) -> dict[int, float]:
    """ranks: 1-based rank of the (first) valid target for each query, or None
    if the target was absent from the ranking (e.g. missing from the gallery)."""
    n = len(ranks)
    if n == 0:
        return {k: float("nan") for k in k_values}
    return {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in k_values}


def recall_at_k_by_category(
    ranks: list[int | None],
    categories: list[str],
    k_values: list[int],
) -> dict[str, dict[int, float]]:
    by_cat: dict[str, list[int | None]] = defaultdict(list)
    for r, c in zip(ranks, categories):
        by_cat[c].append(r)
    result = {cat: recall_at_k(rs, k_values) for cat, rs in by_cat.items()}
    result["overall"] = recall_at_k(ranks, k_values)
    return result
