"""
Phase 3 - cosine similarity retrieval over a cached gallery.

No training, no learned parameters: gallery and query embeddings are
assumed already L2-normalized, so cosine similarity reduces to a dot
product (score = q . z).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RetrievalResult:
    ranked_ids: list[str]   # gallery ids sorted by descending score
    scores: torch.Tensor    # [K] scores for ranked_ids, same order


def cosine_scores(query_embeddings: torch.Tensor, gallery_embeddings: torch.Tensor) -> torch.Tensor:
    """[Q, D] x [N, D] -> [Q, N]. Assumes both are already L2-normalized."""
    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings.unsqueeze(0)
    return query_embeddings @ gallery_embeddings.T


def retrieve_top_k(
    query_embedding: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_ids: list[str],
    k: int | None = None,
) -> RetrievalResult:
    """Rank the full gallery for a single query embedding ([D] or [1, D]).

    k=None returns the full ranking (needed for Recall@K where K can equal
    the gallery size); pass an int to keep only the top-k for storage.
    """
    scores = cosine_scores(query_embedding, gallery_embeddings).squeeze(0)  # [N]
    order = torch.argsort(scores, descending=True)
    if k is not None:
        order = order[:k]
    ranked_ids = [gallery_ids[i] for i in order.tolist()]
    return RetrievalResult(ranked_ids=ranked_ids, scores=scores[order])


def rank_of_target(ranked_ids: list[str], target_ids: set[str]) -> int | None:
    """1-based rank of the first matching target id, or None if absent from ranked_ids."""
    for i, gid in enumerate(ranked_ids, start=1):
        if gid in target_ids:
            return i
    return None