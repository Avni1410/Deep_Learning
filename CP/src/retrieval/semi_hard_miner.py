"""
Phase 8 - semi-hard hard-negative mining.

Corrects a specific, diagnosed failure mode in the Phase 6 miner
(src/retrieval/hard_negative_miner.py): that miner selected the top-K
highest-similarity candidates regardless of whether they ranked above or
below the true positive. The Phase 6 audit showed negative_1 similarity
exceeded positive_similarity in 95.7-96.9% of queries across all three
categories - meaning Phase 6 was training against the model's own top-1
RANKING ERRORS, not near-miss "hard but fair" negatives. This is a known
failure mode in metric learning (naive hard-negative mining from an
undertrained encoder yields "impossible" triplets that destabilize
training - Schroff et al., FaceNet, 2015), consistent with the uniform
small regression observed across R@1 through R@50.

Semi-hard mining fixes this by construction: a candidate is only eligible
as a negative if its similarity to the query is STRICTLY LESS than the
positive's similarity (optionally by a configurable margin buffer). Among
eligible candidates, the hardest (highest similarity) are selected - still
genuinely difficult, but never a case where the model already (wrongly)
prefers the negative over the truth.

This module reuses (does not duplicate) Phase 6's duplicate-group
exclusion logic from src/retrieval/hard_negative_miner.py, and fixes a
performance issue there: that module's mine_for_query() reloads the
duplicate-lookup CSV via pandas on EVERY call. Here the lookup is loaded
ONCE by the caller (see mine_phase8_hard_negatives.py) and passed in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.retrieval.cosine_retrieval import retrieve_top_k
from src.retrieval.hard_negative_miner import duplicate_exclusion_ids


@dataclass
class SemiHardMinedNegatives:
    query_id: str
    category: str
    target: str
    candidate: str
    negative_ids: list[str]
    negative_similarities: list[float]
    negative_categories: list[str]
    positive_similarity: float
    strategy: str
    k_retrieval: int
    n_requested: int
    n_found: int  # may be < n_requested if too few semi-hard candidates exist
    shortfall: int = field(init=False)

    def __post_init__(self) -> None:
        self.shortfall = self.n_requested - self.n_found


def mine_semi_hard_for_query(
    query_embedding: torch.Tensor,
    query_id: str,
    category: str,
    target: str,
    candidate: str,
    gallery_ids: list[str],
    gallery_embeddings: torch.Tensor,
    gallery_categories: list[str] | None,
    asin_to_group: dict[tuple[str, str], str],
    group_to_asins: dict[tuple[str, str], set[str]],
    k_retrieval: int,
    n_hard: int,
    strategy: str,  # "similarity" (unrestricted) or "category_similarity"
    margin_buffer: float = 0.0,
) -> SemiHardMinedNegatives:
    """
    margin_buffer: require negative_similarity <= positive_similarity -
    margin_buffer (0.0 = strictly below positive; a small positive value,
    e.g. 0.02, enforces a stricter safety margin at the cost of fewer
    eligible candidates per query).
    """
    if strategy == "category_similarity":
        if gallery_categories is None:
            raise ValueError("category_similarity strategy requires gallery_categories (a combined gallery).")
        keep = [i for i, c in enumerate(gallery_categories) if c == category]
        ids = [gallery_ids[i] for i in keep]
        embeds = gallery_embeddings[keep]
        id_to_category = {gid: category for gid in ids}
    elif strategy == "similarity":
        ids = gallery_ids
        embeds = gallery_embeddings
        if gallery_categories is None:
            id_to_category = {gid: category for gid in ids}
        else:
            id_to_category = dict(zip(gallery_ids, gallery_categories))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    try:
        pos_idx = gallery_ids.index(target)
        positive_similarity = float((query_embedding * gallery_embeddings[pos_idx]).sum().item())
    except ValueError:
        positive_similarity = float("nan")

    excluded = duplicate_exclusion_ids(
        category=category, target=target, candidate=candidate,
        asin_to_group=asin_to_group, group_to_asins=group_to_asins,
    )

    # Retrieve a generous pool: duplicate filtering AND the semi-hard
    # similarity-below-positive filter can both remove candidates, so we
    # need more headroom than plain top-K mining did.
    retrieval_k = min(max(k_retrieval, 4 * n_hard) + len(excluded), len(ids))
    result = retrieve_top_k(query_embedding, embeds, ids, k=retrieval_k)

    threshold = (positive_similarity - margin_buffer) if positive_similarity == positive_similarity else float("inf")
    # (the `!=` self-check above is a NaN guard: if positive_similarity is NaN,
    # we don't want every candidate to fail the comparison silently - instead
    # we disable the semi-hard filter for this query and fall back to plain
    # top-similarity selection, flagged via n_found < n_requested if relevant.)

    filtered_ids: list[str] = []
    filtered_scores: list[float] = []
    filtered_categories: list[str] = []
    selected_duplicate_groups: set[tuple[str, str]] = set()

    for gid, score in zip(result.ranked_ids, result.scores.tolist()):
        if gid in excluded:
            continue
        if score >= threshold:
            continue  # NOT semi-hard: this candidate already outranks (or ties) the positive

        item_category = id_to_category.get(gid, category)
        group_id = asin_to_group.get((item_category, gid))
        if group_id is not None:
            group_key = (item_category, group_id)
            if group_key in selected_duplicate_groups:
                continue
            selected_duplicate_groups.add(group_key)

        filtered_ids.append(gid)
        filtered_scores.append(float(score))
        filtered_categories.append(item_category)
        if len(filtered_ids) >= n_hard:
            break

    return SemiHardMinedNegatives(
        query_id=query_id, category=category, target=target, candidate=candidate,
        negative_ids=filtered_ids, negative_similarities=filtered_scores,
        negative_categories=filtered_categories, positive_similarity=positive_similarity,
        strategy=strategy, k_retrieval=k_retrieval, n_requested=n_hard, n_found=len(filtered_ids),
    )
