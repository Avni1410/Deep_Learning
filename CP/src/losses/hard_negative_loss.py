"""
Phase 6 - explicit hard-negative contrastive loss.

Extends (does not duplicate) the Phase 4/5 contrastive formulation in
src/losses/contrastive.py. That loss uses only in-batch negatives (the
other targets in the same minibatch). This loss adds an EXPLICIT set of
mined hard negatives per query, kept as a separate term (memory-safe
approach per spec Sec 19 - merging all candidates into one set is the
"if memory allows" path, not taken here given the 4GB GPU).

L_HN for query i with positive z+_i and M hard negatives z-_i,1..M:
  s+ = q_i . z+_i / tau
  sj- = q_i . z-_i,j / tau
  L_HN_i = -log[ exp(s+) / (exp(s+) + sum_j exp(sj-)) ]

Implemented via F.cross_entropy (log_softmax + nll) for numerical
stability, equivalent to the logsumexp form the spec asks for.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def hard_negative_contrastive_loss(
    query_embeddings: torch.Tensor,      # [B, D], L2-normalized
    positive_embeddings: torch.Tensor,   # [B, D], L2-normalized
    hard_negative_embeddings: torch.Tensor,  # [B, M, D], L2-normalized
    temperature: float = 0.07,
) -> torch.Tensor:
    B, D = query_embeddings.shape
    M = hard_negative_embeddings.shape[1]
    if hard_negative_embeddings.shape[0] != B or hard_negative_embeddings.shape[2] != D:
        raise ValueError(
            f"hard_negative_embeddings shape {tuple(hard_negative_embeddings.shape)} "
            f"does not match query batch [B={B}, *, D={D}]"
        )

    s_pos = (query_embeddings * positive_embeddings).sum(dim=-1, keepdim=True)     # [B, 1]
    s_neg = torch.bmm(hard_negative_embeddings, query_embeddings.unsqueeze(-1)).squeeze(-1)  # [B, M]

    logits = torch.cat([s_pos, s_neg], dim=1) / temperature  # [B, 1+M]
    labels = torch.zeros(B, dtype=torch.long, device=query_embeddings.device)  # positive is always index 0
    return F.cross_entropy(logits, labels)