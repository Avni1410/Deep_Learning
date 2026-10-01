"""
Phase 6 - offline hard-negative mining.

Workflow:
  1. Encode a gallery.
  2. Encode composed training queries.
  3. Retrieve top-K candidates via cosine similarity.
  4. Remove:
       - the true target ID
       - the query's reference/candidate ID
       - every ASIN that is an exact-image duplicate of the target
       - every ASIN that is an exact-image duplicate of the candidate
  5. Prevent multiple ASINs from the same exact-image duplicate group
     from being selected as separate negatives.
  6. Take the top N_hard remaining candidates as hard negatives.

Duplicate handling is used ONLY during hard-negative mining.

FashionIQ itself is NOT modified and duplicate images are NOT globally
removed from the dataset.

Duplicate lookup:
    experiments/phase6/fashioniq_duplicate_lookup.csv

Expected columns:
    duplicate_group_id
    category
    asin
    md5
    group_size
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from src.retrieval.cosine_retrieval import retrieve_top_k


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DUPLICATE_LOOKUP = (
    REPO_ROOT
    / "experiments"
    / "phase6"
    / "fashioniq_duplicate_lookup.csv"
)


@dataclass
class MinedNegatives:
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


def load_duplicate_lookup(
    lookup_path: Path = DEFAULT_DUPLICATE_LOOKUP,
) -> tuple[
    dict[tuple[str, str], str],
    dict[tuple[str, str], set[str]],
]:
    """
    Load FashionIQ exact-image duplicate groups.

    Returns
    -------
    asin_to_group:
        Maps (category, ASIN) -> duplicate_group_id

    group_to_asins:
        Maps (category, duplicate_group_id) -> all ASINs
        belonging to that duplicate group.
    """

    lookup_path = Path(lookup_path)

    if not lookup_path.exists():
        raise FileNotFoundError(
            f"Duplicate lookup not found:\n{lookup_path}"
        )

    df = pd.read_csv(lookup_path)

    required_columns = {
        "duplicate_group_id",
        "category",
        "asin",
        "md5",
        "group_size",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Duplicate lookup is missing columns: "
            f"{sorted(missing)}"
        )

    asin_to_group: dict[tuple[str, str], str] = {}
    group_to_asins: dict[tuple[str, str], set[str]] = {}

    for row in df.itertuples(index=False):

        category = str(row.category)
        asin = str(row.asin)
        group_id = str(row.duplicate_group_id)

        asin_to_group[(category, asin)] = group_id

        group_key = (category, group_id)

        if group_key not in group_to_asins:
            group_to_asins[group_key] = set()

        group_to_asins[group_key].add(asin)

    return asin_to_group, group_to_asins


def duplicate_exclusion_ids(
    category: str,
    target: str,
    candidate: str,
    asin_to_group: dict[tuple[str, str], str],
    group_to_asins: dict[tuple[str, str], set[str]],
) -> set[str]:
    """
    Return all ASINs that must be excluded for this query.

    Always excludes:
        - target
        - candidate

    Additionally excludes:
        - every ASIN in target's exact-image duplicate group
        - every ASIN in candidate's exact-image duplicate group
    """

    excluded = {
        target,
        candidate,
    }

    for asin in (target, candidate):

        group_id = asin_to_group.get(
            (category, asin)
        )

        if group_id is None:
            continue

        group_key = (
            category,
            group_id,
        )

        excluded.update(
            group_to_asins.get(
                group_key,
                set(),
            )
        )

    return excluded


def _positive_similarity(
    query_embedding: torch.Tensor,
    target: str,
    gallery_ids: list[str],
    gallery_embeddings: torch.Tensor,
) -> float:
    """
    Return similarity(q, true target).

    Returns NaN if the target does not exist in
    the supplied gallery.
    """

    try:
        idx = gallery_ids.index(target)

    except ValueError:
        return float("nan")

    target_embed = gallery_embeddings[idx]

    return float(
        (
            query_embedding * target_embed
        ).sum().item()
    )


def mine_for_query(
    query_embedding: torch.Tensor,
    query_id: str,
    category: str,
    target: str,
    candidate: str,
    gallery_ids: list[str],
    gallery_embeddings: torch.Tensor,
    gallery_categories: list[str] | None,
    k_retrieval: int,
    n_hard: int,
    strategy: str,
) -> MinedNegatives:

    # ---------------------------------------------------------------
    # Select mining gallery
    # ---------------------------------------------------------------

    if strategy == "category_similarity":

        if gallery_categories is None:
            raise ValueError(
                "category_similarity strategy requires "
                "gallery_categories (a combined gallery)."
            )

        keep = [
            i
            for i, c in enumerate(gallery_categories)
            if c == category
        ]

        ids = [
            gallery_ids[i]
            for i in keep
        ]

        embeds = gallery_embeddings[keep]

        id_to_category = {
            gid: category
            for gid in ids
        }

    elif strategy == "similarity":

        ids = gallery_ids
        embeds = gallery_embeddings

        if gallery_categories is None:

            id_to_category = {
                gid: category
                for gid in ids
            }

        else:

            id_to_category = dict(
                zip(
                    gallery_ids,
                    gallery_categories,
                )
            )

    else:

        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    # ---------------------------------------------------------------
    # Positive similarity
    # ---------------------------------------------------------------

    positive_similarity = _positive_similarity(
        query_embedding=query_embedding,
        target=target,
        gallery_ids=gallery_ids,
        gallery_embeddings=gallery_embeddings,
    )

    # ---------------------------------------------------------------
    # Load exact-image duplicate groups
    # ---------------------------------------------------------------

    asin_to_group, group_to_asins = (
        load_duplicate_lookup()
    )

    # Exclude:
    #   target
    #   candidate
    #   duplicates of target
    #   duplicates of candidate
    excluded = duplicate_exclusion_ids(
        category=category,
        target=target,
        candidate=candidate,
        asin_to_group=asin_to_group,
        group_to_asins=group_to_asins,
    )

    # ---------------------------------------------------------------
    # Retrieve extra candidates.
    #
    # We retrieve more than k_retrieval where possible because
    # duplicate filtering may remove candidates.
    # ---------------------------------------------------------------

    retrieval_k = min(
        k_retrieval + len(excluded),
        len(ids),
    )

    result = retrieve_top_k(
        query_embedding,
        embeds,
        ids,
        k=retrieval_k,
    )

    # ---------------------------------------------------------------
    # Select negatives.
    #
    # IMPORTANT:
    # A duplicate group can contain multiple ASINs representing
    # exactly the same image.
    #
    # We therefore allow at most ONE negative from each duplicate
    # group.
    #
    # Because retrieve_top_k returns candidates in descending
    # similarity order, the first ASIN from a duplicate group is
    # the hardest representative of that group.
    # ---------------------------------------------------------------

    filtered_ids: list[str] = []
    filtered_scores: list[float] = []
    filtered_categories: list[str] = []

    selected_duplicate_groups: set[
        tuple[str, str]
    ] = set()

    for gid, score in zip(
        result.ranked_ids,
        result.scores.tolist(),
    ):

        # -----------------------------------------------------------
        # Target / candidate / their duplicate groups
        # -----------------------------------------------------------

        if gid in excluded:
            continue

        # -----------------------------------------------------------
        # Identify the duplicate group of this candidate.
        # -----------------------------------------------------------

        group_id = asin_to_group.get(
            (
                id_to_category.get(
                    gid,
                    category,
                ),
                gid,
            )
        )

        if group_id is not None:

            group_key = (
                id_to_category.get(
                    gid,
                    category,
                ),
                group_id,
            )

            # -------------------------------------------------------
            # If another ASIN from the same exact-image group has
            # already been selected, skip this candidate.
            # -------------------------------------------------------

            if group_key in selected_duplicate_groups:
                continue

            selected_duplicate_groups.add(
                group_key
            )

        # -----------------------------------------------------------
        # Accept this hard negative.
        # -----------------------------------------------------------

        filtered_ids.append(gid)

        filtered_scores.append(
            float(score)
        )

        filtered_categories.append(
            id_to_category.get(
                gid,
                "unknown",
            )
        )

        # -----------------------------------------------------------
        # Stop once enough negatives have been collected.
        # -----------------------------------------------------------

        if len(filtered_ids) >= n_hard:
            break

    return MinedNegatives(
        query_id=query_id,
        category=category,
        target=target,
        candidate=candidate,
        negative_ids=filtered_ids,
        negative_similarities=filtered_scores,
        negative_categories=filtered_categories,
        positive_similarity=positive_similarity,
        strategy=strategy,
        k_retrieval=k_retrieval,
    )


def random_negatives_for_query(
    query_id: str,
    category: str,
    target: str,
    candidate: str,
    gallery_ids: list[str],
    n_hard: int,
    rng,
    gallery_categories: list[str] | None = None,
) -> MinedNegatives:
    """
    Random-negative baseline.

    Random negatives respect:
        - target exclusion
        - candidate exclusion
        - target duplicate-group exclusion
        - candidate duplicate-group exclusion

    Additionally, at most one ASIN from each duplicate group
    is selected.
    """

    asin_to_group, group_to_asins = (
        load_duplicate_lookup()
    )

    excluded = duplicate_exclusion_ids(
        category=category,
        target=target,
        candidate=candidate,
        asin_to_group=asin_to_group,
        group_to_asins=group_to_asins,
    )

    # ---------------------------------------------------------------
    # Build category lookup.
    # ---------------------------------------------------------------

    if gallery_categories is None:

        id_to_category = {
            gid: category
            for gid in gallery_ids
        }

    else:

        id_to_category = dict(
            zip(
                gallery_ids,
                gallery_categories,
            )
        )

    # ---------------------------------------------------------------
    # Build valid random pool.
    # ---------------------------------------------------------------

    pool = [
        gid
        for gid in gallery_ids
        if gid not in excluded
    ]

    # ---------------------------------------------------------------
    # Randomly shuffle through candidates while respecting
    # duplicate groups.
    # ---------------------------------------------------------------

    rng.shuffle(pool)

    chosen: list[str] = []

    selected_duplicate_groups: set[
        tuple[str, str]
    ] = set()

    for gid in pool:

        if len(chosen) >= n_hard:
            break

        item_category = id_to_category.get(
            gid,
            category,
        )

        group_id = asin_to_group.get(
            (
                item_category,
                gid,
            )
        )

        if group_id is not None:

            group_key = (
                item_category,
                group_id,
            )

            if group_key in selected_duplicate_groups:
                continue

            selected_duplicate_groups.add(
                group_key
            )

        chosen.append(gid)

    categories = [
        id_to_category.get(
            gid,
            "unknown",
        )
        for gid in chosen
    ]

    return MinedNegatives(
        query_id=query_id,
        category=category,
        target=target,
        candidate=candidate,
        negative_ids=chosen,
        negative_similarities=[
            float("nan")
        ] * len(chosen),
        negative_categories=categories,
        positive_similarity=float("nan"),
        strategy="random",
        k_retrieval=0,
    )