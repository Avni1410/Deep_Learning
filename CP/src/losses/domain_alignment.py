"""
Phase 5 - cross-domain representation alignment losses.

Both assume inputs are [B, D] visual embeddings (not necessarily
pre-normalized here - paired_cosine_alignment_loss normalizes internally
via cosine_similarity; coral_loss operates on raw feature covariance).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def paired_cosine_alignment_loss(f_user: torch.Tensor, f_shop: torch.Tensor) -> torch.Tensor:
    """L_pair = mean(1 - cos(f_user_i, f_shop_i)) over the batch."""
    return (1.0 - F.cosine_similarity(f_user, f_shop, dim=-1)).mean()


def coral_loss(user_feats: torch.Tensor, shop_feats: torch.Tensor) -> torch.Tensor:
    """L_CORAL = ||C_user - C_shop||_F^2 / (4 d^2).

    Covariance needs at least 2 samples per side; with batch_size < 2 on
    either side (e.g. a ragged final batch) this returns 0 rather than NaN
    from a division by zero - documented, not silently wrong.
    NOTE: the original Deep CORAL paper's /(4d^2) normalizer assumes
    unnormalized deep-activation scale. For L2-normalized, unit-sphere
    features (as used throughout this project), that constant crushes any
    physically plausible covariance difference to ~1e-7 regardless of batch
    size (diagnosed empirically on real DeepFashion2 features at B=2..64;
    see docs/phase_5_cross_domain_alignment.md). Dividing by d instead keeps
    the same covariance-difference formulation while remaining at a scale
    comparable to L_pair for this project's normalized feature space.
    """
    B_u, d = user_feats.shape
    B_s, _ = shop_feats.shape
    if B_u < 2 or B_s < 2:
        return torch.zeros((), device=user_feats.device, dtype=user_feats.dtype)
    u = user_feats - user_feats.mean(dim=0, keepdim=True)
    s = shop_feats - shop_feats.mean(dim=0, keepdim=True)
    cov_u = (u.T @ u) / (B_u - 1)
    cov_s = (s.T @ s) / (B_s - 1)
    diff = cov_u - cov_s
    return (diff * diff).sum() / d


def combined_domain_loss(
    f_user: torch.Tensor, f_shop: torch.Tensor, alpha: float = 1.0, beta: float = 0.1
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l_pair = paired_cosine_alignment_loss(f_user, f_shop)
    l_coral = coral_loss(f_user, f_shop)
    total = alpha * l_pair + beta * l_coral
    return total, {"pair": l_pair, "coral": l_coral}