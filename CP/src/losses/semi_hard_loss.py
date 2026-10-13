"""
Phase 8 - hard-negative contrastive loss for VARIABLE negative counts.

Semi-hard mining (src/retrieval/semi_hard_miner.py) can find fewer than
N_hard eligible negatives for some queries (when few gallery items rank
below the positive similarity) - this is reported as `shortfall` in the
mined CSV, not silently padded with an invalid negative. This loss
function accepts a per-sample list of negative embeddings (length may
differ per query, including 0) and computes the InfoNCE-style loss only
over samples that have at least one real negative; samples with zero
eligible negatives still receive the ordinary CIR loss (computed
separately), just not this hard-negative term.

For query i with positive z+_i and M_i (possibly 0) hard negatives:
  s+ = q_i . z+_i / tau
  sj- = q_i . z-_i,j / tau  for j in 1..M_i
  L_i = -log[ exp(s+) / (exp(s+) + sum_j exp(sj-)) ]   if M_i > 0, else skipped

Returns the mean over samples with M_i > 0, and the count of such samples
(so a caller can log what fraction of the batch actually contributed).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def hard_negative_loss_variable(
    query_embeddings: torch.Tensor,            # [B, D], L2-normalized
    positive_embeddings: torch.Tensor,         # [B, D], L2-normalized
    negative_embeddings_list: list[torch.Tensor | None],  # length B, each [M_i, D] or None/empty
    temperature: float = 0.07,
) -> tuple[torch.Tensor, int]:
    B = query_embeddings.shape[0]
    if len(negative_embeddings_list) != B:
        raise ValueError(f"negative_embeddings_list has length {len(negative_embeddings_list)}, expected {B}")

    losses = []
    for i in range(B):
        negs = negative_embeddings_list[i]
        if negs is None or negs.shape[0] == 0:
            continue
        q = query_embeddings[i : i + 1]           # [1, D]
        pos = positive_embeddings[i : i + 1]      # [1, D]
        s_pos = (q * pos).sum(dim=-1, keepdim=True)        # [1, 1]
        s_neg = (negs @ q.T).T                             # [1, M_i]
        logits = torch.cat([s_pos, s_neg], dim=1) / temperature
        label = torch.zeros(1, dtype=torch.long, device=q.device)
        losses.append(F.cross_entropy(logits, label))

    if not losses:
        return torch.zeros((), device=query_embeddings.device, dtype=query_embeddings.dtype), 0
    return torch.stack(losses).mean(), len(losses)
