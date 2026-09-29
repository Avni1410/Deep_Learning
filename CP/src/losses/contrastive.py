"""
Phase 4 - symmetric query<->target contrastive loss.

s_ij = q_i . z_j / tau
L_q->t = CE(s, diag)   ;   L_t->q = CE(s.T, diag)
L_CIR = 0.5 * (L_q->t + L_t->q)

q and z are assumed already L2-normalized.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_contrastive_loss(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    B = query_embeddings.shape[0]
    logits = query_embeddings @ target_embeddings.T / temperature   # [B, B]
    labels = torch.arange(B, device=query_embeddings.device)
    loss_q2t = F.cross_entropy(logits, labels)
    loss_t2q = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_q2t + loss_t2q)